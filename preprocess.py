import json
import re
import os

def clean_medical_text(text):
    if not text:
        return ""

    text = re.sub(r"(Best wishes|Sincerely|Thanks for choosing|Chat Doctor|MD|Dr\.)[\s\w,]*", "", text, flags=re.IGNORECASE)
    
    text = re.sub(r"(Hi|Hello|Dear)\s+[A-Z][a-z]+", r"\1 Patient", text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_data():
    raw_path = 'data/raw/'
    processed_path = 'data/processed/'
    os.makedirs(processed_path, exist_ok=True)
    
    final_data = []

    hc_file = os.path.join(raw_path, 'HealthCareMagic-100k.json')
    if os.path.exists(hc_file):
        print(f"Lendo {hc_file}...")
        with open(hc_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data[:10000]:
                final_data.append({
                    "instruction": "Responda à consulta médica de forma profissional, informativa e empática.",
                    "input": clean_medical_text(item.get('input', '')),
                    "output": clean_medical_text(item.get('output', ''))
                })
        print(f"-> 10.000 registros processados do HealthCareMagic.")
    else:
        print(f"AVISO: Arquivo {hc_file} não encontrado.")

    ic_file = os.path.join(raw_path, 'iCliniq.json')
    if os.path.exists(ic_file):
        print(f"Lendo {ic_file}...")
        with open(ic_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data[:5000]:
                final_data.append({
                    "instruction": "Aja como um assistente médico virtual. Analise os sintomas e sugira condutas clínicas baseadas em protocolos.",
                    "input": clean_medical_text(item.get('input', '')),
                    "output": clean_medical_text(item.get('answer_chatdoctor', ''))
                })
        print(f"-> 5.000 registros processados do iCliniq.")
    else:
        print(f"AVISO: Arquivo {ic_file} não encontrado.")

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