import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def chat_with_model(model_dir: str):
    print("Loading fine-tuned model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    print("Model and tokenizer loaded successfully.")

    print("Starting chat session. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chat session.")
            break

        # Prepend special tokens to user input
        formatted_input = f"<|input|>{user_input}<|endofinput|>"

        # Tokenize user input with explicit max_length and attention_mask
        inputs = tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512  # Explicitly set max_length
        )
        print(f"Tokenized input: {inputs}")

        # Generate response with attention_mask and adjusted decoding parameters
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Explicitly pass attention_mask
            max_new_tokens=50,  # Limit the number of new tokens generated
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.7,  # Add randomness to the output
            top_p=0.9,       # Use nucleus sampling
            top_k=50,        # Limit to top-k tokens
            do_sample=True   # Enable sampling-based generation
        )
        print(f"Generated token IDs: {outputs}")

        # Decode and print the response, removing special tokens
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Decoded response: {response}")

        # Extract the response after the <|output|> token
        if "<|output|>" in response:
            response = response.split("<|output|>")[1].split("<|endofoutput|>")[0].strip()
        print(f"Model: {response}")

if __name__ == "__main__":
    # Path to the fine-tuned model directory
    print('Initializing chat...')
    model_directory = os.path.join(os.path.dirname(__file__), "fine_tuned_model")
    if not os.path.exists(model_directory):
        print(f"Model directory {model_directory} does not exist.")
    else:
        print(f"Model directory {model_directory} exists.")
        # Start the chat session with the fine-tuned model
        print('Starting chat...')
        chat_with_model(model_directory)
