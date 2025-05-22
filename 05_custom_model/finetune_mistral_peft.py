import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Step 1: Load and format your dataset
def format_prompts(examples):
    # Adjust this function to match your dataset structure
    # Here we assume 'prompt' and 'completion' keys as in your previous data
    texts = []
    for prompt, completion in zip(examples['prompt'], examples['completion']):
        texts.append(f"[INST] {prompt.strip()} [/INST] {completion.strip()}")
    return {'text': texts}

dataset = load_dataset("json", data_files={"train": "train-data.json"})
dataset = dataset["train"].map(format_prompts, batched=True)

# Step 2: Set up the model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# Step 3: Set up PEFT (LoRA)
config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "lm_head"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# Step 4: Training arguments
args = TrainingArguments(
    output_dir="mistral-7b-instruct-lora",
    num_train_epochs=4,
    per_device_train_batch_size=2,  # Adjust for your GPU
    learning_rate=1e-5,
    optim="adamw_torch"
)

# Step 5: Trainer
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length=1024,
)

trainer.train()

# Step 6: Merge LoRA adapter and model
adapter_model = trainer.model
merged_model = adapter_model.merge_and_unload()
trained_tokenizer = tokenizer

# Step 7: (Optional) Push to Hugging Face Hub
# repo_id = "your_hf_username/mistral-7b-instruct-lora"
# merged_model.push_to_hub(repo_id)
# trained_tokenizer.push_to_hub(repo_id)
