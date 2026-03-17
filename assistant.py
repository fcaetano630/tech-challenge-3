import os
import torch
import logging
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Importações para o LangGraph (Requisito do Edital)
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# 1. Configurações Iniciais e Logging (Requisito: Auditoria)
load_dotenv()

logging.basicConfig(
    filename='auditoria_medica.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# --- CONFIGURAÇÃO DE HARDWARE ---
# Se a RTX 5080 der erro de "Kernel", mude para False para usar o Ryzen 9800X3D
USE_GPU = True 

if USE_GPU:
    device_map = "auto"
    dtype_config = torch.float16
else:
    device_map = {"": "cpu"}
    dtype_config = torch.float32 

# Definição do Estado para o LangGraph
class AgentState(TypedDict):
    pergunta: str
    contexto: str
    resposta: str
    fontes: List[str]

def start_assistant():
    print(f"\n--- 🏥 Assistente Médico (Hardware: {'GPU' if USE_GPU else 'CPU'}) ---")
    
    base_model_id = "unsloth/llama-3-8b-bnb-4bit"
    adapter_path = "model" 
    records_dir = "data/records"

    # 2. Carregar Modelo e seu Treinamento
    print("🤖 Carregando cérebro da IA...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype_config,
            device_map=device_map,
            trust_remote_code=True
        )
        print("🔧 Aplicando treinamento especializado (Fine-tuning)...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
    except Exception as e:
        print(f"❌ Erro no carregamento: {e}")
        return

    # 3. Preparação do RAG
    print("📄 Indexando prontuários locais...")
    if not os.path.exists(records_dir): os.makedirs(records_dir)

    # Loader configurado com UTF-8 para evitar erros de acentuação
    loader = DirectoryLoader(
        records_dir, 
        glob="*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    
    documents = loader.load()

    if not documents:
        print("⚠️ Aviso: Nenhum prontuário encontrado.")
        vectorstore = None
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        print(f"✅ {len(documents)} prontuários indexados!")

    # --- NÓS DO LANGGRAPH ---

    def no_recuperador(state: AgentState):
        """Busca as informações no prontuário (RAG)"""
        pergunta = state['pergunta']
        fontes = []
        contexto = "Nenhum histórico localizado."
        
        if vectorstore:
            docs = vectorstore.similarity_search(pergunta, k=2)
            contexto = "\n".join([d.page_content for d in docs])
            fontes = list(set([os.path.basename(d.metadata['source']) for d in docs]))
            
        return {"contexto": contexto, "fontes": fontes}

    def no_gerador(state: AgentState):
        """Gera a resposta focando na extração do nome do paciente"""
        # Prompt otimizado para não ignorar o nome do paciente
        prompt = f"""Você é um assistente médico brasileiro de alta precisão.
Responda SEMPRE em Português (Brasil).

INSTRUÇÃO:
1. Identifique o NOME do paciente citado no prontuário abaixo.
2. Responda à pergunta usando EXCLUSIVAMENTE os dados fornecidos.
3. Se a pergunta for "Quem tem X doença", localize o nome no início do texto.

CONHECIMENTO DO PRONTUÁRIO:
{state['contexto']}

PERGUNTA:
{state['pergunta']}

RESPOSTA MÉDICA (Seja direto e cite o nome do paciente):"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if USE_GPU else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=300,
                temperature=0.2, # Leve aumento para melhorar a associação de ideias
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        res_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        res_limpa = res_full.split("RESPOSTA MÉDICA (Seja direto e cite o nome do paciente):")[-1].strip()
        
        aviso = "\n\n⚠️ Apoio à decisão clínica. Validar conduta com um médico responsável."
        
        return {"resposta": res_limpa + aviso}

    # CONSTRUÇÃO DO GRAFO
    workflow = StateGraph(AgentState)
    workflow.add_node("recuperar", no_recuperador)
    workflow.add_node("gerar", no_gerador)
    workflow.set_entry_point("recuperar")
    workflow.add_edge("recuperar", "gerar")
    workflow.add_edge("gerar", END)
    app = workflow.compile()

    # 5. Interface
    print("\n🩺 Assistente pronto!")
    while True:
        msg = input("\nVocê: ")
        if msg.lower() in ['sair', 'exit', 'quit']: break
        
        try:
            resultado = app.invoke({"pergunta": msg})
            print(f"\nIA Médica: {resultado['resposta']}")
            print(f"📚 Fontes: {', '.join(resultado['fontes'])}")
            
            logging.info(f"Q: {msg} | Fontes: {resultado['fontes']} | A: {resultado['resposta']}")
            
        except Exception as e:
            print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    start_assistant()