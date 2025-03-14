"""script to demonstrate ollama simple chatbox"""
from libs.assistant_helper import Assistant

# Initialize the Ollama client


# Function to interact with the chatbot
class MyChatbot():
    """Custom chatbot class"""
    def __init__(self):
        self.assistant = Assistant('mistral', None)

    def ask_question(self, question):
        """Custom chatbot custom initiator"""
        # Get the assistant's response
        assistant_response = self.assistant.generate(
            prompt=question
        )

        # Extract the assistant's message
        print(f" assistant: {assistant_response}")

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
