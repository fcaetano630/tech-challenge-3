import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env (onde deve estar o HF_TOKEN)
load_dotenv()

def run_fine_tuning():
    # 1. Autenticação Segura
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Aviso: HF_TOKEN não encontrado no arquivo .env. Certifique-se de que ele existe para baixar o modelo base.")

    # Configurações de ambiente para GPU e Gerenciamento de Memória
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    max_seq_length = 2048
    model_name = "unsloth/llama-3-8b-bnb-4bit"

    # 2. Carregar Modelo e Tokenizer
    print("Carregando modelo base quantizado...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )

    # 3. Configuração do LoRA (Adapters treináveis)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
    )

    # 4. Preparação do Prompt (Padrão Alpaca)
    alpaca_prompt = """Abaixo está uma instrução que descreve uma tarefa médica. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{}

### Entrada:
{}

### Resposta:
{}"""

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
            texts.append(text)
        return { "text" : texts, }

    # 5. Carregamento do Dataset Processado
    dataset_path = "data/processed/medical_train_dataset.jsonl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Arquivo não encontrado em: {dataset_path}. Execute o preprocess.py primeiro.")

    print("Carregando e mapeando dataset...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 6. Configuração do Treinador (Parâmetros validados no sucesso do Colab)
    print("Configurando Treinador...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = TrainingArguments(
            per_device_train_batch_size = 1,      # Configuração para evitar OOM
            gradient_accumulation_steps = 8,      # Total batch size = 8
            warmup_steps = 40,                    # 10% do treino
            max_steps = 400,                      # O alvo final que gerou o adapter
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",                 # Otimizador eficiente em memória
            weight_decay = 0.01,
            output_dir = "outputs",
            save_strategy = "no",
        ),
    )

    # 7. Execução
    print("Iniciando treinamento especializado...")
    torch.cuda.empty_cache() # Limpa resíduos da GPU antes de começar
    trainer.train()

    # 8. Salvamento do Adapter Final
    output_dir = "model/medical_llama3_adapter"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Sucesso! O modelo médico especializado foi salvo em: {output_dir}")

if __name__ == "__main__":
    run_fine_tuning()