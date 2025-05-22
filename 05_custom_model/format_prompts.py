import json

def format_prompts(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted = []
    for item in data:
        if "prompt" in item and "completion" in item:
            text = f"[INST] {item['prompt'].strip()} [/INST] {item['completion'].strip()}"
            formatted.append({"text": text})
        elif "instruction" in item and "response" in item:
            text = f"[INST] {item['instruction'].strip()} [/INST] {item['response'].strip()}"
            formatted.append({"text": text})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "train-data.json"
    output_file = "formatted-train-data.json"
    format_prompts(input_file, output_file)
