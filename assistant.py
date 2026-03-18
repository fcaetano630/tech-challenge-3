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
# Requisito do Edital: LangGraph
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# 1. Configurações de Auditoria
load_dotenv()
logging.basicConfig(
    filename='auditoria_medica.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# --- CONFIGURAÇÃO DE HARDWARE ---
USE_GPU = True 

if USE_GPU:
    device_map = "auto"
    dtype_config = torch.float16
else:
    device_map = {"": "cpu"}
    dtype_config = torch.float32 

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

    # 2. Carregar Modelo + Seu Fine-tuning (800 gerações)
    print("🤖 Carregando cérebro da IA...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype_config,
            device_map=device_map,
            trust_remote_code=True
        )
        print(f"🔧 Aplicando adaptador de treino: {adapter_path}")
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
        # Aumentamos o chunk para garantir que o nome do paciente não seja separado do diagnóstico
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        print(f"✅ {len(documents)} prontuários indexados!")

    # --- NÓS DO LANGGRAPH ---

    def no_recuperador(state: AgentState):
        """Busca os fatos no prontuário (RAG)"""
        pergunta = state['pergunta']
        fontes = []
        contexto = "Informação não localizada nos registros locais."
        
        if vectorstore:
            docs = vectorstore.similarity_search(pergunta, k=2)
            contexto = "\n".join([d.page_content for d in docs])
            fontes = list(set([os.path.basename(d.metadata['source']) for d in docs]))
            
        return {"contexto": contexto, "fontes": fontes}

    def no_gerador(state: AgentState):
        """Gera a resposta focando no contexto fornecido"""
        
        # PROMPT DE ALTA ATENÇÃO: Força o modelo a olhar para o contexto do RAG
        prompt = f"""### Instrução:
Você é o médico responsável. Analise o PRONTUÁRIO abaixo e responda à pergunta identificando claramente o paciente. 
Não use informações de outros pacientes ou conhecimentos gerais que conflitem com este documento.

### PRONTUÁRIO PARA ANÁLISE:
{state['contexto']}

### PERGUNTA DO COLEGA MÉDICO:
{state['pergunta']}

### RESPOSTA MÉDICA (Identifique o Paciente e o Caso):"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if USE_GPU else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=400,
                temperature=0.3, # Reduzido para maior fidelidade aos dados do prontuário
                top_p=0.85,
                repetition_penalty=1.2, # Evita que ele 'vicie' em frases do treino
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        res_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        res_limpa = res_full.split("### RESPOSTA MÉDICA (Identifique o Paciente e o Caso):")[-1].strip()
        
        aviso = "\n\n---\n⚠️ [Apoio à decisão clínica. Validar conduta com médico responsável.]"
        
        return {"resposta": res_limpa + aviso}

    # CONSTRUÇÃO DO GRAFO (LangGraph)
    workflow = StateGraph(AgentState)
    workflow.add_node("recuperar", no_recuperador)
    workflow.add_node("gerar", no_gerador)
    workflow.set_entry_point("recuperar")
    workflow.add_edge("recuperar", "gerar")
    workflow.add_edge("gerar", END)
    app = workflow.compile()

    # 5. Loop de Atendimento
    print("\n🩺 Assistente pronto! (Digite 'sair')")
    while True:
        msg = input("\nPergunta: ")
        if msg.lower() in ['sair', 'exit', 'quit']: break
        
        try:
            resultado = app.invoke({"pergunta": msg})
            
            print(f"\nIA: {resultado['resposta']}")
            if resultado['fontes']:
                print(f"📚 Fonte: {', '.join(resultado['fontes'])}")
            
            logging.info(f"Q: {msg} | Fontes: {resultado['fontes']} | A: {resultado['resposta']}")
            
        except Exception as e:
            print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    start_assistant()