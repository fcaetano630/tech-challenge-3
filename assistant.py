import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configurações Iniciais
load_dotenv()

# --- CONFIGURAÇÃO DE HARDWARE ---
# Mude para True para usar GPU (se compatível) ou False para usar seu Ryzen 9800X3D
USE_GPU = True 

# Configuração dinâmica baseada na escolha acima
if USE_GPU:
    device_map = "auto"
    dtype_config = torch.float16
else:
    device_map = {"": "cpu"}
    dtype_config = torch.float32 # Essencial para não travar na CPU
# --------------------------------

def start_assistant():
    print(f"\n--- 🏥 Assistente Médico (Hardware: {'GPU' if USE_GPU else 'CPU'}) ---")
    
    # Caminhos originais que você estava usando
    base_model_id = "unsloth/llama-3-8b-bnb-4bit"
    adapter_path = "model" 
    records_dir = "data/records"

    # 2. Carregar Modelo e seu Treinamento (Adapter)
    print("🤖 Carregando cérebro da IA (isso pode levar um minuto)...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype_config,
            device_map=device_map,
            trust_remote_code=True
        )
        
        # Carrega o seu Fine-tuning
        print("🔧 Aplicando treinamento especializado...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model.eval()
    except Exception as e:
        print(f"❌ Erro no carregamento: {e}")
        return

    # 3. Preparação do RAG (Leitura dos Prontuários)
    print("📄 Indexando prontuários locais...")
    if not os.path.exists(records_dir): os.makedirs(records_dir)

    loader = DirectoryLoader(records_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("⚠️ Aviso: Nenhum prontuário encontrado em data/records/.")
        vectorstore = None
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        # Embeddings sempre em CPU para estabilidade
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        print("✅ Prontuários indexados com FAISS!")

    # 4. Lógica de Resposta
    def responder(pergunta):
        contexto = ""
        if vectorstore:
            docs = vectorstore.similarity_search(pergunta, k=2)
            contexto = "\n".join([d.page_content for d in docs])

        prompt = f"""Você é um assistente médico. Use o prontuário para ser preciso. 

CONHECIMENTO DO PRONTUÁRIO:
{contexto}

PERGUNTA DO USUÁRIO:
{pergunta}

RESPOSTA MÉDICA:"""

        # Direciona os inputs para o hardware correto
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if USE_GPU else "cpu")
        
        print("⏳ IA processando a resposta...")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=250,
                temperature=0.7,
                do_sample=True,
                use_cache=True, # Garante que ele consiga "reler" e manter o fluxo
                pad_token_id=tokenizer.eos_token_id
            )
        
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return res.split("RESPOSTA MÉDICA:")[-1].strip()

    # 5. Interface de Chat
    print("\n🩺 Assistente pronto! (ou 'sair' para encerrar)")
    while True:
        msg = input("\nVocê: ")
        if msg.lower() in ['sair', 'exit', 'quit']: break
        
        print("IA: Analisando dados...")
        try:
            print(f"\nIA Médica: {responder(msg)}")
        except Exception as e:
            print(f"\n❌ Erro ao gerar resposta: {e}")

if __name__ == "__main__":
    start_assistant()