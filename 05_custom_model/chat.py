from transformers import AutoModelForCausalLM, AutoTokenizer

model_directory = os.path.join(os.path.dirname(__file__), "fine_tuned_model")

tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)

device = "cpu" # the device to load the model onto

messages = [
    {"role": "user", "content": "What is your favourite condiment?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
