```markdown
# Ollama Chatbot with Context

This script demonstrates how to build a chatbot using the Ollama framework with a system prompt and conversation context. The chatbot is tailored to assist with cloud DevOps-related questions and provide troubleshooting advice.

## Features

- **Custom System Prompt**: The chatbot is initialized with a system prompt that sets its role as an assistant specializing in cloud DevOps.
- **Contextual Conversation**: The chatbot maintains conversation history to provide better responses by considering previous interactions.
- **Interactive CLI**: Users can interact with the chatbot via the command line interface (CLI).

## Prerequisites

- Python 3.x or later
- [Ollama](https://ollama.com/download) installed on your system
- A downloaded model (`llama3.2`):
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

3. Download the model (`llama3.2`) using Ollama:
   ```bash
   ollama pull llama3.2
   ```

## Usage

1. Run the chatbot script:
   ```bash
   python 03_ollama_chatbot_with_context/ollama_chatbot_with_context.py
   ```

2. The chatbot will prompt you to enter a question. Type your question, and the assistant will respond. Type `exit` to end the session.

## License

This project is licensed under the MIT License.
```

This README captures the main functionality of the script, including how to set up, run, and interact with the chatbot, along with any necessary prerequisites and installation steps.