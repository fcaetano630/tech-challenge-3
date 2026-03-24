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

# 1. Configurações de Auditoria (Requisito: Rastreabilidade)
load_dotenv()
logging.basicConfig(filename='auditoria.log', level=logging.INFO, format='%(asctime)s - %(message)s')

USE_GPU = torch.cuda.is_available()
device = "cuda" if USE_GPU else "cpu"
dtype_config = torch.bfloat16 if (USE_GPU and torch.cuda.is_bf16_supported()) else torch.float16

class AgentState(TypedDict):
    pergunta: str
    contexto: str
    resposta: str
    fontes: List[str]

def start_assistant():
    base_model_id = "unsloth/llama-3-8b-bnb-4bit"
    adapter_path = "model" 
    records_dir = "data/records"

    # 2. Carregamento do Modelo (ETAPA: LLM & LoRA Adapter)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            torch_dtype=dtype_config, 
            device_map="auto" if USE_GPU else None
        )
        
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
    except Exception as e:
        print(f"Erro: {e}"); return

    # 3. Indexação (ETAPA: Recuperação de Informação)
    if not os.path.exists(records_dir): os.makedirs(records_dir)
    loader = DirectoryLoader(records_dir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        print(f"RAG: {len(documents)} documentos na base vetorial.")
    else:
        vectorstore = None

    # --- LÓGICA DO AGENTE (ETAPA: Orquestração LangGraph) ---

    def no_recuperador(state: AgentState):
        contexto = ""
        fontes = []
        if vectorstore:
            docs = vectorstore.similarity_search(state['pergunta'], k=2)
            for d in docs:
                origem = os.path.basename(d.metadata['source'])
                contexto += f"[FONTE OFICIAL {origem}]: {d.page_content}\n"
                fontes.append(origem)
        return {"contexto": contexto, "fontes": list(set(fontes))}

    def no_gerador(state: AgentState):
        prompt = f"""### Medical Database:
{state['contexto']}

### Task:
Answer the question using the database above. Be extremely short.

### Question: {state['pergunta']}

### Response (Starting with "Based on records,"): Based on records,"""

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                repetition_penalty=1.2, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        res_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        res_limpa = res_full.split("Response (Starting with \"Based on records,\"):")[-1].strip()

        return {"resposta": res_limpa}

    # 4. Grafo de Estados
    workflow = StateGraph(AgentState)
    workflow.add_node("recuperar", no_recuperador)
    workflow.add_node("gerar", no_gerador)
    workflow.set_entry_point("recuperar")
    workflow.add_edge("recuperar", "gerar")
    workflow.add_edge("gerar", END)
    app = workflow.compile()

    while True:
        msg = input("\nAuditoria > ")
        if msg.lower() in ['sair', 'exit']: break
        try:
            resultado = app.invoke({"pergunta": msg})
            print(f"\nANÁLISE EXTRAÍDA:\n{resultado['resposta']}")
            print(f"FONTES UTILIZADAS: {resultado['fontes']}")
        except Exception as e:
            print(f"Erro: {e}")

if __name__ == "__main__":
    start_assistant()