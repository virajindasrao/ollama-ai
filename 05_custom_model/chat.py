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

        # Tokenize user input with explicit max_length and attention_mask
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512  # Explicitly set max_length
        )

        # Generate response with attention_mask
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Explicitly pass attention_mask
            max_length=512,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # Decode and print the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Model: {response}")

if __name__ == "__main__":
    # Path to the fine-tuned model directory
    print('initializing chat')
    print('loading model')  
    model_directory = os.path.join(os.path.dirname(__file__), "fine_tuned_model")
    if not os.path.exists(model_directory):
        print(f"Model directory {model_directory} does not exist.")
    else:
        print(f"Model directory {model_directory} exists.")
        # Start the chat session with the fine-tuned model
        print('starting chat')
        chat_with_model(model_directory)
