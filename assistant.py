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

load_dotenv()
logging.basicConfig(filename='auditoria_medica.log', level=logging.INFO, format='%(asctime)s - %(message)s', encoding='utf-8')

USE_GPU = True 
device_map = "auto" if USE_GPU else {"": "cpu"}
dtype_config = torch.float16 if USE_GPU else torch.float32

class AgentState(TypedDict):
    pergunta: str
    contexto: str
    resposta: str
    fontes: List[str]

def start_assistant():
    print(f"\n--- 🏥 Assistente de Prontuários (Hardware: {'GPU' if USE_GPU else 'CPU'}) ---")
    
    base_model_id = "unsloth/llama-3-8b-bnb-4bit"
    adapter_path = "model" 
    records_dir = "data/records"

    # 1. Carregamento do Cérebro
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=dtype_config, device_map=device_map, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
    except Exception as e:
        print(f"❌ Erro: {e}"); return

    # 2. Indexação com Foco em Identidade
    if not os.path.exists(records_dir): os.makedirs(records_dir)
    loader = DirectoryLoader(records_dir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()

    if not documents:
        print("⚠️ Pasta de registros vazia."); vectorstore = None
    else:
        # Chunks maiores para não separar Nome de Diagnóstico
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        vectorstore = FAISS.from_documents(texts, embeddings)
        print(f"✅ {len(documents)} prontuários carregados.")

    # --- NÓS DO GRAFO ---

    def no_recuperador(state: AgentState):
        pergunta = state['pergunta']
        contexto_final = ""
        fontes = []
        
        if vectorstore:
            # Buscamos 3 blocos para garantir que o nome apareça
            docs = vectorstore.similarity_search(pergunta, k=3)
            for d in docs:
                nome_arquivo = os.path.basename(d.metadata['source'])
                # Injetamos o nome do arquivo no texto para a IA não se perder
                contexto_final += f"\n[DOCUMENTO: {nome_arquivo}]\n{d.page_content}\n"
                fontes.append(nome_arquivo)
            
        return {"contexto": contexto_final, "fontes": list(set(fontes))}

    def no_gerador(state: AgentState):
        # Prompt Ultra-Direcionado
        prompt = f"""### Instrução:
Você é um assistente de auditoria médica. Sua tarefa é ler o prontuário e responder perguntas sobre o paciente específico.
Não invente nomes. Se o nome não estiver no prontuário abaixo, diga que não encontrou.

### PRONTUÁRIOS ENCONTRADOS:
{state['contexto']}

### PERGUNTA:
{state['pergunta']}

### RESPOSTA (Cite o nome do paciente):"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if USE_GPU else "cpu")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=350,
                temperature=0.2, # Baixa temperatura = Menos "alucinação" do treino
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        res_limpa = res.split("### RESPOSTA (Cite o nome do paciente):")[-1].strip()
        return {"resposta": res_limpa + "\n\n[Validação Médica Obrigatória]"}

    # Grafo
    workflow = StateGraph(AgentState)
    workflow.add_node("recuperar", no_recuperador)
    workflow.add_node("gerar", no_gerador)
    workflow.set_entry_point("recuperar")
    workflow.add_edge("recuperar", "gerar")
    workflow.add_edge("gerar", END)
    app = workflow.compile()

    print("\n🩺 Pronto para consultas!")
    while True:
        msg = input("\nPergunta: ")
        if msg.lower() in ['sair', 'exit']: break
        try:
            resultado = app.invoke({"pergunta": msg})
            print(f"\nIA: {resultado['resposta']}")
            print(f"📚 Arquivos lidos: {resultado['fontes']}")
            logging.info(f"Q: {msg} | F: {resultado['fontes']}")
        except Exception as e: print(f"❌ Erro: {e}")

if __name__ == "__main__":
    start_assistant()