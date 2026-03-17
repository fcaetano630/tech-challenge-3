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

# 1. Configurações Iniciais e Auditoria
load_dotenv()

logging.basicConfig(
    filename='auditoria_medica.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# --- CONFIGURAÇÃO DE HARDWARE (RTX 5080 ou CPU) ---
USE_GPU = True 

if USE_GPU:
    device_map = "auto"
    dtype_config = torch.float16
else:
    device_map = {"": "cpu"}
    dtype_config = torch.float32 

# Definição do Estado do Agente
class AgentState(TypedDict):
    pergunta: str
    contexto: str
    resposta: str
    fontes: List[str]

def start_assistant():
    print(f"\n--- 🏥 Assistente Médico Híbrido (Hardware: {'GPU' if USE_GPU else 'CPU'}) ---")
    
    base_model_id = "unsloth/llama-3-8b-bnb-4bit"
    adapter_path = "model" 
    records_dir = "data/records"

    # 2. Carregar Modelo + Fine-tuning
    print("🤖 Carregando cérebro da IA...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype_config,
            device_map=device_map,
            trust_remote_code=True
        )
        print("🔧 Aplicando seu treinamento especializado...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
    except Exception as e:
        print(f"❌ Erro no carregamento: {e}")
        return

    # 3. Preparação do RAG (Consulta de Dados)
    print("📄 Indexando prontuários locais (UTF-8)...")
    if not os.path.exists(records_dir): os.makedirs(records_dir)

    loader = DirectoryLoader(
        records_dir, 
        glob="*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    
    documents = loader.load()

    if not documents:
        print("⚠️ Aviso: Pasta data/records vazia.")
        vectorstore = None
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        print(f"✅ {len(documents)} arquivos prontos para consulta!")

    # --- NÓS DO LANGGRAPH ---

    def no_recuperador(state: AgentState):
        """Busca fatos no prontuário para apoiar a IA"""
        pergunta = state['pergunta']
        fontes = []
        contexto = "Nenhum histórico específico encontrado nos arquivos locais."
        
        if vectorstore:
            docs = vectorstore.similarity_search(pergunta, k=2)
            contexto = "\n".join([d.page_content for d in docs])
            fontes = list(set([os.path.basename(d.metadata['source']) for d in docs]))
            
        return {"contexto": contexto, "fontes": fontes}

    def no_gerador(state: AgentState):
        """Gera a resposta usando o modelo treinado + contexto extraído"""
        # Formato de instrução otimizado para o Llama-3 brilhar
        prompt = f"""### Instrução:
Você é um médico assistente altamente qualificado. Sua tarefa é responder à pergunta do usuário de forma profissional, clara e ética. 
Utilize o contexto do prontuário abaixo se ele for relevante, mas use também seu conhecimento clínico especializado para formular a melhor resposta.

### Contexto do Prontuário Local:
{state['contexto']}

### Pergunta do Usuário:
{state['pergunta']}

### Resposta Médica (em Português):"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if USE_GPU else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=400,
                temperature=0.7, # Aumentado para permitir o uso do conhecimento do Fine-tuning
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        res_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extrai apenas a parte da resposta após o cabeçalho
        res_limpa = res_full.split("### Resposta Médica (em Português):")[-1].strip()
        
        # Guardrail de Segurança (Obrigatório)
        aviso = "\n\n---\n[Apoio à decisão clínica. Esta análise deve ser validada por um profissional médico.]"
        
        return {"resposta": res_limpa + aviso}

    # CONSTRUÇÃO DO FLUXO (LangGraph)
    workflow = StateGraph(AgentState)
    workflow.add_node("recuperar", no_recuperador)
    workflow.add_node("gerar", no_gerador)
    workflow.set_entry_point("recuperar")
    workflow.add_edge("recuperar", "gerar")
    workflow.add_edge("gerar", END)
    app = workflow.compile()

    # 5. Interface de Chat
    print("\n🩺 Assistente pronto! (Digite 'sair' para encerrar)")
    while True:
        msg = input("\nVocê: ")
        if msg.lower() in ['sair', 'exit', 'quit']: break
        
        try:
            # Execução via LangGraph
            resultado = app.invoke({"pergunta": msg})
            
            print(f"\nIA Médica: {resultado['resposta']}")
            
            # Exibe as fontes para cumprir o requisito de Explainability
            if resultado['fontes']:
                print(f"📚 Fontes consultadas: {', '.join(resultado['fontes'])}")
            
            # Auditoria em log
            logging.info(f"Pergunta: {msg} | Fontes: {resultado['fontes']} | Resposta: {resultado['resposta']}")
            
        except Exception as e:
            print(f"\n❌ Erro no processamento: {e}")

if __name__ == "__main__":
    start_assistant()