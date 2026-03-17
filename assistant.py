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
# Mude para False se a sua GPU (Série 50) ainda apresentar erro de Kernel
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
            dtype=dtype_config, # Atualizado de torch_dtype para dtype
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
        print(f"❌ Erro no carregamento do modelo: {e}")
        return

    # 3. Preparação do RAG (Requisito: Consulta em base de dados)
    print("📄 Indexando prontuários locais...")
    if not os.path.exists(records_dir): os.makedirs(records_dir)

    # CORREÇÃO DE ENCODING: Adicionado loader_kwargs para ler acentos (UTF-8)
    loader = DirectoryLoader(
        records_dir, 
        glob="*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    
    try:
        documents = loader.load()
    except Exception as e:
        print(f"❌ Erro ao ler arquivos: {e}")
        return

    if not documents:
        print("⚠️ Aviso: Nenhum prontuário encontrado em data/records/.")
        vectorstore = None
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        # Embeddings (Processamento de linguagem natural)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        print(f"✅ {len(documents)} prontuários indexados com sucesso!")

    # --- NÓS DO LANGGRAPH ---

    def no_recuperador(state: AgentState):
        """Busca as informações no prontuário (RAG)"""
        pergunta = state['pergunta']
        fontes = []
        contexto = "Nenhum histórico localizado no prontuário."
        
        if vectorstore:
            # Requisito: Explainability (Indicar a fonte)
            docs = vectorstore.similarity_search(pergunta, k=2)
            contexto = "\n".join([d.page_content for d in docs])
            fontes = list(set([os.path.basename(d.metadata['source']) for d in docs]))
            
        return {"contexto": contexto, "fontes": fontes}

    def no_gerador(state: AgentState):
        """Gera a resposta em Português"""
        prompt = f"""Você é um assistente médico brasileiro. Responda em Português (Brasil).
Use o CONHECIMENTO DO PRONTUÁRIO abaixo para responder à pergunta de forma precisa.
Se a informação não estiver no prontuário, responda que não encontrou dados nos registros.

CONHECIMENTO DO PRONTUÁRIO:
{state['contexto']}

PERGUNTA:
{state['pergunta']}

RESPOSTA MÉDICA EM PORTUGUÊS:"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if USE_GPU else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=300,
                temperature=0.1, 
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        res_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Limpa a resposta para exibir apenas o que a IA criou
        res_limpa = res_full.split("RESPOSTA MÉDICA EM PORTUGUÊS:")[-1].strip()
        
        # Requisito: Segurança (Nunca prescrever sem validação humana)
        aviso_legal = "\n\n⚠️ IMPORTANTE: Esta é uma análise de apoio à decisão clínica. Toda conduta e prescrição deve ser validada por um médico responsável."
        
        return {"resposta": res_limpa + aviso_legal}

    # CONSTRUÇÃO DO FLUXO (LangGraph)
    workflow = StateGraph(AgentState)
    workflow.add_node("recuperar", no_recuperador)
    workflow.add_node("gerar", no_gerador)
    workflow.set_entry_point("recuperar")
    workflow.add_edge("recuperar", "gerar")
    workflow.add_edge("gerar", END)
    app = workflow.compile()

    # 5. Interface de Chat
    print("\n🩺 Assistente pronto! Digite sua pergunta ou 'sair'.")
    while True:
        msg = input("\nMédico: ")
        if msg.lower() in ['sair', 'exit', 'quit']: break
        
        try:
            # Invoca o grafo de estados
            resultado = app.invoke({"pergunta": msg})
            
            print(f"\nIA Médica: {resultado['resposta']}")
            
            # Requisito: Explicabilidade (Exibe os arquivos consultados)
            if resultado['fontes']:
                print(f"📚 Fontes: {', '.join(resultado['fontes'])}")
            
            # Requisito: Auditoria (Salva no log)
            logging.info(f"Pergunta: {msg} | Fontes: {resultado['fontes']} | Resposta: {resultado['resposta']}")
            
        except Exception as e:
            print(f"\n❌ Erro durante o processamento: {e}")

if __name__ == "__main__":
    start_assistant()