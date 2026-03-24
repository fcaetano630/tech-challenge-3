import torch
from unsloth import FastLanguageModel
import os
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 0. Prevenção de erro de memória (OOM)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# 1. Configurações do Modelo
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 2. Configuração do LoRA (Ajustado para maior densidade técnica)
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
)

# 3. Preparação do Prompt (Refinado para Persona Médica)
alpaca_prompt = """Você é um assistente médico especializado. Responda à instrução abaixo com precisão técnica e tom profissional.

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

dataset = load_dataset("json", data_files="medical_train_dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. Configuração do Treinador (AJUSTADO PARA QUALIDADE E PRECISÃO)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 100,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        output_dir = "outputs",
        save_strategy = "steps",
        save_steps = 500,
    ),
)

# 5. INICIAR TREINAMENTO
print("Iniciando treinamento focado em precisão médica...")
torch.cuda.empty_cache()
trainer.train()

# 6. SALVAR NA PASTA 'model' (Para o assistant.py reconhecer)
save_directory = "model"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
