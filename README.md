It seems that I'm unable to directly access the repository structure. However, based on your request, I can guide you through generating a `README.md` for the repository.

Here’s a general template for your `README.md` based on the typical contents of the repository:

---

# Ollama AI

Ollama AI is a collection of sample projects and experiments showcasing how to build chatbots and integrate AI using the Ollama framework. This repository includes various scripts that demonstrate different chatbot functionalities, from simple implementations to more advanced configurations with custom system prompts and conversation context.

## Features

- **Simple Chatbot**: A basic chatbot implementation using the Ollama framework.
- **Custom System Prompts**: Chatbots configured with specific roles or instructions for consistent behavior.
- **Context-Aware Chatbots**: Examples where the chatbot remembers and continues conversations.
- **Interactive CLI**: Users can interact with the chatbot through the command line.

## Directory Structure

```
ollama-ai/
├── 01_ollama_simple_chatbot/
│   └── 01_ollama_simple_chatbot.py
├── 02_ollama_chatbot_with_custom_system_prompt/
│   └── 02_ollama_chatbot_with_custom_system_prompt.py
├── 03_ollama_chatbot_with_context/
│   └── 03_ollama_chatbot_with_context.py
├── requirements.txt
└── README.md
```

## Prerequisites

Before running any of the scripts, ensure that you have the following installed:

- Python 3.x or later
- [Ollama](https://ollama.com/download) installed on your system
- A downloaded model (`llama3.2` or another model supported by Ollama):
  - Run `ollama pull llama3.2` to download the model.
  - Confirm the download by running `ollama show llama3.2`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/virajindasrao/ollama-ai.git
   cd ollama-ai
   ```

2. Install any required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the model (`llama3.2` or other models supported by Ollama):
   ```bash
   ollama pull llama3.2
   ```

## Usage

Each folder contains a Python script showcasing a specific feature. Here are the instructions for each:

- **01_ollama_simple_chatbot**: Run the simple chatbot script:
   ```bash
   python 01_ollama_simple_chatbot/01_ollama_simple_chatbot.py
   ```

- **02_ollama_chatbot_with_custom_system_prompt**: Run the chatbot with a custom system prompt:
   ```bash
   python 02_ollama_chatbot_with_custom_system_prompt/02_ollama_chatbot_with_custom_system_prompt.py
   ```

- **03_ollama_chatbot_with_context**: Run the chatbot that maintains conversation context:
   ```bash
   python 03_ollama_chatbot_with_context/03_ollama_chatbot_with_context.py
   ```

## License

This project is licensed under the MIT License.

---

This template assumes that the repository contains multiple examples, such as simple chatbots and ones with context handling, based on the previous scripts you've shared. Adjust the specifics as needed to match the actual structure and features of the repository.