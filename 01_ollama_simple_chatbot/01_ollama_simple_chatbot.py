"""script to demonstrate ollama simple chatbox"""
from libs.assistant_helper import Assistant

class MyChatbot():
    """Custom chatbot class"""
    def __init__(self):
        # The given model should be downloaded on local before calling from the script
        # To download the model run command `ollama pull llama3.2`
        # To confirm the download run command `ollama show llama3.2`
        # To download ollama on local check this and follow the steps : https://ollama.com/download
        self.assistant = Assistant('llama3.2', None)

    def ask_question(self, question):
        """Custom chatbot custom initiator"""
        # Get the assistant's response
        assistant_response = self.assistant.generate(
            prompt=question
        )

        # Extract the assistant's message
        print(f" assistant: {assistant_response}")
        print("******************************************************")

def main():
    """Custom chatbot init function"""
    # Initiate chatbot class
    bot = MyChatbot()
    print("Hi! This is AI assistant. Type your question or exit to close.")
    # loop until user enters 'exit'
    while True:
        content = input("user: ")
        if content == 'exit':
            return 0
        bot.ask_question(content)


if __name__ == "__main__":
    main()
