```markdown
# Ollama Chatbot with Custom System Prompt

This script demonstrates how to create a customized chatbot using the Ollama framework with a specific system prompt. The chatbot is tailored for answering questions about cloud DevOps, providing advice, and troubleshooting problems related to this domain.

## Features

- Customizable system prompt for the chatbot.
- Cloud DevOps-focused assistant that can handle troubleshooting and provide advice.
- Interactive user interface for real-time Q&A.

## Prerequisites

- Python 3.x or later
- [Ollama](https://ollama.com/download) installed on your system
- A downloaded model (`llama3.2`):
  - Run `ollama pull llama3.2` to download the model
  - Confirm the download by running `ollama show llama3.2`

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

3. Download the model (`llama3.2`) using Ollama:
   ```bash
   ollama pull llama3.2
   ```

## Usage

1. Run the chatbot script:
   ```bash
   python 02_ollama_chatbot_with_custom_system_prompt/02_ollama_chatbot_with_custom_system_prompt.py
   ```

2. The chatbot will prompt you to enter a question. Type your question, and the assistant will respond. Type `exit` to end the session.

## License

This project is licensed under the MIT License.
```

This README captures the core functionality of the script, including instructions for setup and usage. You can adjust it based on your specific needs.