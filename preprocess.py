import json
import re
import os

def clean_medical_text(text):
    """
    Realiza a limpeza e anonimização básica do texto médico.
    Remove assinaturas, nomes de doutores conhecidos e saudações.
    """
    if not text:
        return ""
    
    # 1. Remove assinaturas e termos de encerramento comuns (Chat Doctor, MD, Dr. Silva, etc)
    text = re.sub(r"(Best wishes|Sincerely|Thanks for choosing|Chat Doctor|MD|Dr\.)[\s\w,]*", "", text, flags=re.IGNORECASE)
    
    # 2. Anonimização: Substitui nomes próprios após saudações (Hi Mark -> Hi Patient)
    text = re.sub(r"(Hi|Hello|Dear)\s+[A-Z][a-z]+", r"\1 Patient", text)
    
    # 3. Limpeza de espaços extras e quebras de linha desnecessárias
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_data():
    # Definição de caminhos (Seguindo sua estrutura de pastas)
    raw_path = 'data/raw/'
    processed_path = 'data/processed/'
    os.makedirs(processed_path, exist_ok=True)
    
    final_data = []

    # --- 1. Processando HealthCareMagic (Meta: 10.000 exemplos) ---
    hc_file = os.path.join(raw_path, 'HealthCareMagic-100k.json')
    if os.path.exists(hc_file):
        print(f"Lendo {hc_file}...")
        with open(hc_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Aumentado para 10.000 registros
            for item in data[:10000]:
                final_data.append({
                    "instruction": "Responda à consulta médica de forma profissional, informativa e empática.",
                    "input": clean_medical_text(item.get('input', '')),
                    "output": clean_medical_text(item.get('output', ''))
                })
        print(f"-> 10.000 registros processados do HealthCareMagic.")
    else:
        print(f"AVISO: Arquivo {hc_file} não encontrado.")

    # --- 2. Processando iCliniq (Meta: 5.000 exemplos) ---
    ic_file = os.path.join(raw_path, 'iCliniq.json') # Verifique se o nome está com 'Q' maiúsculo ou minúsculo no seu PC
    if os.path.exists(ic_file):
        print(f"Lendo {ic_file}...")
        with open(ic_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Aumentado para 5.000 registros
            for item in data[:5000]:
                # Usamos 'answer_chatdoctor' conforme o padrão profissional do dataset
                final_data.append({
                    "instruction": "Aja como um assistente médico virtual. Analise os sintomas e sugira condutas clínicas baseadas em protocolos.",
                    "input": clean_medical_text(item.get('input', '')),
                    "output": clean_medical_text(item.get('answer_chatdoctor', ''))
                })
        print(f"-> 5.000 registros processados do iCliniq.")
    else:
        print(f"AVISO: Arquivo {ic_file} não encontrado.")

    # --- 3. Salvando o Dataset Unificado ---
    output_file = os.path.join(processed_path, 'medical_train_dataset.jsonl')
    
    print(f"Salvando dataset final em {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in final_data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print("-" * 30)
    print(f"CONCLUÍDO!")
    print(f"Total de exemplos: {len(final_data)}")
    print(f"Local: {output_file}")
    print("-" * 30)

if __name__ == "__main__":
    prepare_data()