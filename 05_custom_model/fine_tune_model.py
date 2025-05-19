import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfFolder
from typing import Tuple
import argparse
import psutil  # Add this import to monitor system resources

from transformers import logging

# Ensure CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU visibility
print("The script will run on CPU only.")

# Set Hugging Face API token from environment variable
HfFolder.save_token(os.environ.get("HUGGINGFACE_TOKEN", ""))

class CustomDataset(Dataset):
    def __init__(self, dataset_file_path: str, tokenizer_obj, max_seq_length: int = 512)::  # Increased max_seq_length
        print("Checking if dataset file exists...")
        if not os.path.exists(dataset_file_path):
            print(f"Dataset file not found: {dataset_file_path}")
            raise FileNotFoundError(f"Dataset file not found: {dataset_file_path}")
        print("Dataset file found. Reading data...")

        with open(dataset_file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        print("Dataset loaded successfully. Preparing tokenizer...")
        
        self.tokenizer = tokenizer_obj
        self.max_length = max_seq_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        # Extract 'prompt' and 'completion' keys
        if 'prompt' in item and 'completion' in item:
            input_text = f"### Instruction:\n{item['prompt']}\n\n### Response:\n"
            target_text = f"{item['completion']}"letion']}<|endofoutput|>"
        else:
            raise KeyError(f"Missing required keys in dataset item at index {idx}. Expected keys: 'prompt' and 'completion'.")

        # Tokenize input and target
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        # Concatenate input and target training
        input_ids = inputs['input_ids'].squeeze(0)
        labels = targets['input_ids'].squeeze(0)

        # Create a new input that is the full prompt + completion others should be -100
        input_ids_full = torch.cat([input_ids, labels], dim=0)re loss on input
        labels_full[-len(labels):] = labels  # Compute loss only on output
        # Create attention mask accordingly
        attention_mask_full = torch.ones_like(input_ids_full)
            'input_ids': input_ids,
        # Create labels: only the target part should have labels; others should be -100
        labels_full = torch.cat([
            torch.full_like(input_ids, -100),  # ignore loss on input
            labels  # compute loss only on output
        ], dim=0)
def load_model_and_tokenizer(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        # Truncate if needed and model...")
        max_len = self.max_length
        input_ids_full = input_ids_full[:max_len]ined(model_id)
        attention_mask_full = attention_mask_full[:max_len]
        labels_full = labels_full[:max_len]
        # Add special tokens if not already present
        return {tokens = {"additional_special_tokens": ["<|input|>", "<|endofinput|>", "<|output|>", "<|endofoutput|>"]}
            'input_ids': input_ids_full,(special_tokens)
            'attention_mask': attention_mask_full,)
            'labels': labels_full
        } Set pad_token to eos_token if not already set
        if tokenizer_obj.pad_token is None:
            tokenizer_obj.pad_token = tokenizer_obj.eos_token
def load_model_and_tokenizer(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    print("Loading tokenizer and model...")
    try:print(f"Error loading tokenizer: {e}")
        tokenizer_obj = AutoTokenizer.from_pretrained(model_id)
        print(f"Tokenizer loaded successfully: {model_id}")
    try:
        # Add special tokens if not already presentained(model_id)
        special_tokens = {"additional_special_tokens": ["<|input|>", "<|endofinput|>", "<|output|>", "<|endofoutput|>"]}
        tokenizer_obj.add_special_tokens(special_tokens)
        print("Special tokens added to tokenizer.")
        print(f"Error loading model: {e}")
        # Set pad_token to eos_token if not already set
        if tokenizer_obj.pad_token is None:
            tokenizer_obj.pad_token = tokenizer_obj.eos_token
            print("pad_token was not set. Using eos_token as pad_token.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise_obj,
    train_data: Dataset,
    try:ut_folder: str,
        model_obj = AutoModelForCausalLM.from_pretrained(model_id)
        model_obj.resize_token_embeddings(len(tokenizer_obj))  # Resize embeddings for new tokens
        print("Model loaded successfully.")
    except Exception as e:"checkpoints"
        print(f"Error loading model: {e}")
        raisearting fine-tuning process...")
    device = torch.device("cpu")  # Force CPU-only execution
    return tokenizer_obj, model_obj)

def fine_tune_model(ing DataLoader...")
    model_obj,er = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    tokenizer_obj,ader initialized with batch size: {batch_size}")
    train_data: Dataset,
    output_folder: str, optimizer...")
    epochs: int = 5,  # Increased epochs for better trainingh.optim.AdamW(model_obj.parameters(), lr=learning_rate)
    batch_size: int = 4,  # Reduced batch size for better gradient updatesitialized with learning rate: {learning_rate}")
    learning_rate: float = 3e-5,  # Adjusted learning rate
    checkpoint_dir: str = "checkpoints"oding gradients
):  gradient_clip_value = 1.0
    print("Starting fine-tuning process...")
    device = torch.device("cpu")  # Force CPU-only execution
    print(f"Using device: {device}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Initializing DataLoader...")kpoint_dir, "checkpoint.pth")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    print(f"DataLoader initialized with batch size: {batch_size}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Initializing optimizer...")kpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model_obj.parameters(), lr=learning_rate)
    print(f"Optimizer initialized with learning rate: {learning_rate}")
        print(f"Resuming training from epoch {start_epoch}.")
    # Gradient clipping to prevent exploding gradients
    gradient_clip_value = 1.0found. Starting training from scratch.")

    # Load checkpoint if availableng mode...")
    start_epoch = 0()
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path): 1}/{epochs}...")
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_obj.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])r)}...")
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}.")
    else:   input_ids = batch['input_ids'].to(device)
        print("No checkpoint found. Starting training from scratch.")
            labels = batch['labels'].to(device)
    print("Setting model to training mode...")
    model_obj.train()adients
            optimizer.zero_grad()
    for epoch in range(start_epoch, epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}...")
        total_loss = 0model_obj(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        for batch_idx, batch in enumerate(train_loader):
            print(f"Processing batch {batch_idx + 1}/{len(train_loader)}...")
            if torch.isnan(loss):
            # Move input data to devicetered at batch {batch_idx + 1}. Skipping this batch.")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss.backward()
            # Zero gradients
            optimizer.zero_grad()revent exploding gradients
            torch.nn.utils.clip_grad_norm_(model_obj.parameters(), gradient_clip_value)
            # Forward pass
            outputs = model_obj(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Check for NaN loss
            if torch.isnan(loss):em()
                print(f"NaN loss encountered at batch {batch_idx + 1}. Skipping this batch.")
                continuess for every 10 batches
            if (batch_idx + 1) % 10 == 0:
            # Backward passch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            loss.backward()
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
            # Clip gradients to prevent exploding gradientsge Loss: {avg_loss:.4f}")
            torch.nn.utils.clip_grad_norm_(model_obj.parameters(), gradient_clip_value)
        # Save checkpoint
            # Optimizer steppoint for epoch {epoch + 1}...")
            optimizer.step() = checkpoint_path + ".tmp"
        try:
            # Accumulate loss
            total_loss += loss.item()
                'model_state_dict': model_obj.state_dict(),
            # Log progress for every 10 batcheser.state_dict(),
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            os.replace(temp_checkpoint_path, checkpoint_path)
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        print(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
            print(f"Error saving checkpoint: {e}")
        # Save checkpointists(temp_checkpoint_path):
        print(f"Saving checkpoint for epoch {epoch + 1}...")
        temp_checkpoint_path = checkpoint_path + ".tmp"
        try:ine-tuning process completed.")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_obj.state_dict(),t_folder}...")
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_lossut_folder)
            }, temp_checkpoint_path)tput_folder)
            os.replace(temp_checkpoint_path, checkpoint_path)
            print(f"Checkpoint for epoch {epoch + 1} saved successfully.")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            if os.path.exists(temp_checkpoint_path):
                os.remove(temp_checkpoint_path)n="Fine-tune a Hugging Face model.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    print("Fine-tuning process completed.")=str, required=True, help="Path to the training data")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model", help="Output directory")
    # Save the fine-tuned modelhs", type=int, default=3, help="Number of training epochs")
    print(f"Saving fine-tuned model and tokenizer to {output_folder}...")size")
    os.makedirs(output_folder, exist_ok=True)pe=float, default=5e-5, help="Learning rate")
    model_obj.save_pretrained(output_folder)
    tokenizer_obj.save_pretrained(output_folder)
    print("Model and tokenizer saved successfully.")
    print("Loading tokenizer and model...")
if __name__ == "__main__":odel
    logging.set_verbosity_info()l_and_tokenizer(args.model_name)
    print("Tokenizer and model loaded successfully.")
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    output_folder = os.path.join(os.path.dirname(__file__), 'fine_tuned_model')
    args = parser.parse_args()
    print("Starting fine-tuning process...")
    print("Loading tokenizer and model...")
    # Load tokenizer and model
    tokenizer, model = load_model_and_tokenizer(args.model_name)
    print("Tokenizer and model loaded successfully.")
        train_dataset,
    print("Preparing dataset...")
    # Prepare datasetpochs,
    train_dataset = CustomDataset(args.data_path, tokenizer)
    print("Dataset prepared successfully.")
    )
    # Ensure the output directory is set to 'fine_tuned_model'
    output_folder = os.path.join(os.path.dirname(__file__), 'fine_tuned_model')
    print(f"Saving fine-tuned model and tokenizer to {output_folder}...")
    print("Starting fine-tuning process...")fully.")    # Fine-tune the model    fine_tune_model(        model,        tokenizer,        train_dataset,        output_folder,        epochs=args.epochs,        batch_size=args.batch_size,        learning_rate=args.learning_rate    )    print("Fine-tuning process completed.")    print(f"Saving fine-tuned model and tokenizer to {output_folder}...")    print("Model and tokenizer saved successfully.")