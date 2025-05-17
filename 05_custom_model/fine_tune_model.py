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
    def __init__(self, dataset_file_path: str, tokenizer_obj, max_seq_length: int = 512):
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

        # Handle both 'input'/'output' and 'prompt'/'completion' keys
        if 'input' in item and 'output' in item:
            input_text = item['input']
            target_text = item['output']
        elif 'prompt' in item and 'completion' in item:
            input_text = item['prompt']
            target_text = item['completion']
        else:
            raise KeyError(f"Missing required keys in dataset item at index {idx}. Expected keys: 'input'/'output' or 'prompt'/'completion'.")

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

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': targets['input_ids'].squeeze(0)
        }

def load_model_and_tokenizer(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    print("Loading tokenizer and model...")
    try:
        tokenizer_obj = AutoTokenizer.from_pretrained(model_id)
        print(f"Tokenizer loaded successfully: {model_id}")

        # Set pad_token to eos_token if not already set
        if tokenizer_obj.pad_token is None:
            tokenizer_obj.pad_token = tokenizer_obj.eos_token
            print("pad_token was not set. Using eos_token as pad_token.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

    try:
        model_obj = AutoModelForCausalLM.from_pretrained(model_id)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    return tokenizer_obj, model_obj

def fine_tune_model(
    model_obj,
    tokenizer_obj,
    train_data: Dataset,
    output_folder: str,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    checkpoint_dir: str = "checkpoints"
):
    print("Starting fine-tuning process...")
    device = torch.device("cpu")  # Force CPU-only execution
    print(f"Using device: {device}")

    print("Initializing DataLoader...")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    print(f"DataLoader initialized with batch size: {batch_size}")

    print("Initializing optimizer...")
    optimizer = torch.optim.AdamW(model_obj.parameters(), lr=learning_rate)
    print(f"Optimizer initialized with learning rate: {learning_rate}")

    # Load checkpoint if available
    start_epoch = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        try:
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_obj.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0
    else:
        print("No checkpoint found. Starting training from scratch.")

    print("Setting model to training mode...")
    model_obj.train()

    for epoch in range(start_epoch, epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}...")
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            print(f"Processing batch {batch_idx + 1}/{len(train_loader)}...")

            # Log input data details
            print(f"Batch {batch_idx + 1}: Moving input data to device...")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            print(f"Batch {batch_idx + 1}: Input data moved to device successfully.")
            print(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")

            # Zero gradients
            print(f"Batch {batch_idx + 1}: Zeroing gradients...")
            optimizer.zero_grad()
            print(f"Batch {batch_idx + 1}: Gradients zeroed.")
            print(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")

            # Forward pass
            print(f"Batch {batch_idx + 1}: Performing forward pass...")
            outputs = model_obj(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            print(f"Batch {batch_idx + 1}: Forward pass completed. Loss: {loss.item():.4f}")
            print(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")

            # Backward pass
            print(f"Batch {batch_idx + 1}: Performing backward pass...")
            loss.backward()
            print(f"Batch {batch_idx + 1}: Backward pass completed.")
            print(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")

            # Optimizer step
            print(f"Batch {batch_idx + 1}: Updating model parameters...")
            optimizer.step()
            print(f"Batch {batch_idx + 1}: Model parameters updated.")
            print(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")

            # Accumulate loss
            total_loss += loss.item()
            print(f"Batch {batch_idx + 1}: Loss accumulated. Current total loss: {total_loss:.4f}")
            print(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")

            # Log progress for every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                print(f"CPU Usage: {psutil.cpu_percent()}%, Memory Usage: {psutil.virtual_memory().percent}%")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_loss:.4f}")

        # Save checkpointporary file first
        print(f"Saving checkpoint for epoch {epoch + 1}...")
        temp_checkpoint_path = checkpoint_path + ".tmp"
        try:ict': model_obj.state_dict(),
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_obj.state_dict(),            }, temp_checkpoint_path, _use_new_zipfile_serialization=False, pickle_protocol=4)
                'optimizer_state_dict': optimizer.state_dict(),le with the new one
                'loss': total_loss            os.replace(temp_checkpoint_path, checkpoint_path)
            }, temp_checkpoint_path, _use_new_zipfile_serialization=False, pickle_protocol=4)for epoch {epoch + 1} saved successfully.")
            os.replace(temp_checkpoint_path, checkpoint_path)
            print(f"Checkpoint for epoch {epoch + 1} saved successfully."){e}")
        except Exception as e:t_path):
            print(f"Error saving checkpoint: {e}")
            if os.path.exists(temp_checkpoint_path):
                os.remove(temp_checkpoint_path)    print("Fine-tuning process completed.")

    print("Fine-tuning process completed.")
    print(f"Saving fine-tuned model and tokenizer to {output_folder}...")
    # Save the fine-tuned model
    print(f"Saving fine-tuned model and tokenizer to {output_folder}...")
    os.makedirs(output_folder, exist_ok=True)
    model_obj.save_pretrained(output_folder)
    tokenizer_obj.save_pretrained(output_folder)
    print("Model and tokenizer saved successfully.")

if __name__ == "__main__":
    logging.set_verbosity_info()Parser(description="Fine-tune a Hugging Face model.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model.")=str, required=True, help="Path to the training data")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")put_dir", type=str, default="fine_tuned_model", help="Output directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")umber of training epochs")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model", help="Output directory")ault=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")se_args()

    args = parser.parse_args()
    # Load tokenizer and model
    print("Loading tokenizer and model...")e)
    # Load tokenizer and model
    tokenizer, model = load_model_and_tokenizer(args.model_name)
    print("Tokenizer and model loaded successfully.")

    print("Preparing dataset...")CustomDataset(args.data_path, tokenizer)
    # Prepare datasetaset prepared successfully.")
    train_dataset = CustomDataset(args.data_path, tokenizer)
    print("Dataset prepared successfully.")t directory is set to 'fine_tuned_model'
.path.join(os.path.dirname(__file__), 'fine_tuned_model')
    # Ensure the output directory is set to 'fine_tuned_model'
    output_folder = os.path.join(os.path.dirname(__file__), 'fine_tuned_model')cess...")

    print("Starting fine-tuning process...")ine_tune_model(
    # Fine-tune the model
    fine_tune_model(        tokenizer,
        model,
        tokenizer,










    print("Model and tokenizer saved successfully.")    print(f"Saving fine-tuned model and tokenizer to {output_folder}...")    print("Fine-tuning process completed.")    )        learning_rate=args.learning_rate        batch_size=args.batch_size,        epochs=args.epochs,        output_folder,        train_dataset,        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    print("Fine-tuning process completed.")

    print(f"Saving fine-tuned model and tokenizer to {output_folder}...")
    print("Model and tokenizer saved successfully.")