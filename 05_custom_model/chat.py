import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def chat_with_model(model_dir: str):
    print("Loading fine-tuned model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    print("Model and tokenizer loaded successfully.")

    print("Starting chat session. Type 'exit' to quit.")
    # Maintain chat history for multi-turn conversations
    messages = []
    first_message = True
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chat session.")
            break

        if first_message:
            messages
        # Add user message to chat history
        messages.append({"role": "user", "content": user_input})

        # Use the chat template for encoding
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding="max_length"
        )

        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True
        )
        print(f"Model output (raw): {outputs}")
        # Decode the model output
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)  # <-- Set to False to see <s>
        print(f"Raw model output: {response}")

        # Extract only the assistant's latest response after the last [/INST]
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1]
            # Optionally, stop at the next [INST] if present (for multi-turn)
            if "[INST]" in response:
                response = response.split("[INST]")[0]
        response = response.strip()

        print(f"=====================================")
        print(f"Model: {response}")
        print(f"=====================================")

        # Add assistant response to chat history for context in next turn
        messages.append({"role": "assistant", "content": response})

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
