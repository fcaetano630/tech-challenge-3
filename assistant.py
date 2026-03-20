import os
import torch
import logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# 1. Configurações de Ambiente e Auditoria
load_dotenv()
logging.basicConfig(
    filename='auditoria_medica.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(message)s', 
    encoding='utf-8'
)

# --- CONFIGURAÇÃO DE HARDWARE ---
USE_GPU = True 
device_map = "auto" if USE_GPU else {"": "cpu"}
dtype_config = torch.float16 if USE_GPU else torch.float32

class AgentState(TypedDict):
    pergunta: str
    contexto: str
    resposta: str
    fontes: List[str]

def start_assistant():
    print(f"\n--- 🏥 Assistente de Prontuários v3.0 (Hardware: {'GPU' if USE_GPU else 'CPU'}) ---")
    
    base_model_id = "unsloth/llama-3-8b-bnb-4bit"
    adapter_path = "model" 
    records_dir = "data/records"

    # 2. Carregamento do Modelo com seu Fine-tuning
    print("🤖 Carregando modelo especializado...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            dtype=dtype_config, 
            device_map=device_map, 
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
    except Exception as e:
        print(f"❌ Erro ao carregar IA: {e}"); return

    # 3. Indexação RAG (Calibrada para busca de nomes)
    if not os.path.exists(records_dir): os.makedirs(records_dir)
    
    loader = DirectoryLoader(
        records_dir, 
        glob="*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()

    if not documents:
        print("⚠️ Pasta data/records vazia. Adicione os .txt e reinicie."); vectorstore = None
    else:
        # AJUSTE: Chunks menores (400) para aumentar a precisão da busca por palavras-chave (Nomes)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        print(f"✅ {len(documents)} prontuários indexados com sucesso.")

    # --- FLUXO LANGGRAPH ---

    def no_recuperador(state: AgentState):
        """Busca ampliada para garantir que o paciente correto seja achado"""
        pergunta = state['pergunta']
        contexto_acumulado = ""
        fontes = []
        
        if vectorstore:
            # AJUSTE: k=4 aumenta a chance de achar a Elena se o FAISS estiver 'míope'
            docs = vectorstore.similarity_search(pergunta, k=4)
            
            for d in docs:
                nome_arq = os.path.basename(d.metadata['source'])
                # Injetamos delimitadores fortes para o modelo não misturar os pacientes
                contexto_acumulado += f"\n=== DOCUMENTO: {nome_arq} ===\n"
                contexto_acumulado += f"{d.page_content}\n"
                contexto_acumulado += "============================\n"
                fontes.append(nome_arq)
            
        return {"contexto": contexto_acumulado, "fontes": list(set(fontes))}

    def no_gerador(state: AgentState):
        """Gera a resposta com foco total na identificação do paciente"""
        prompt = f"""### Instrução:
Você é um auditor médico. Analise os prontuários abaixo e responda APENAS sobre o paciente solicitado.
Se houver múltiplos pacientes no contexto, foque apenas naquele que corresponde à pergunta.

### CONTEXTO DOS PRONTUÁRIOS:
{state['contexto']}

### PERGUNTA DO MÉDICO:
{state['pergunta']}

### RESPOSTA MÉDICA (Identifique o nome do paciente):"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if USE_GPU else "cpu")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=400,
                temperature=0.2, # Baixa temperatura para evitar confusão entre arquivos
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        res_limpa = res.split("### RESPOSTA MÉDICA (Identifique o nome do paciente):")[-1].strip()
        
        return {"resposta": res_limpa + "\n\n---\n[Análise de apoio baseada em prontuário local]"}

    # Construção do Grafo
    workflow = StateGraph(AgentState)
    workflow.add_node("recuperar", no_recuperador)
    workflow.add_node("gerar", no_gerador)
    workflow.set_entry_point("recuperar")
    workflow.add_edge("recuperar", "gerar")
    workflow.add_edge("gerar", END)
    app = workflow.compile()

    print("\n🩺 Sistema pronto para consultas!")
    while True:
        msg = input("\nPergunta: ")
        if msg.lower() in ['sair', 'exit', 'quit']: break
        
        try:
            resultado = app.invoke({"pergunta": msg})
            print(f"\nIA Médica: {resultado['resposta']}")
            print(f"📚 Fontes lidas: {resultado['fontes']}")
            
            logging.info(f"Pergunta: {msg} | Fontes: {resultado['fontes']}")
        except Exception as e:
            print(f"❌ Erro no fluxo: {e}")

if __name__ == "__main__":
    start_assistant()