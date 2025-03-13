"""script to demonstrate ollama simple chatbox"""
import ollama

# Initialize the Ollama client


# Function to interact with the chatbot
class MyChatbot():
    """Custom chatbot class"""
    def __init__(self):
        self.client = ollama.Client()

    def ask_question(self, question):
        """Custom chatbot custom initiator"""
        conversation = [
            {
                'role': 'user',
                'content': question
            }
        ]

        # Get the assistant's response
        response = self.client.chat(
            # Ollama model
            model='llama3.2',
            # Conversation message
            messages=conversation,
            # Limit the context window size
            options={'num_ctx': 1024}
        )

        # Extract the assistant's message
        assistant_message = response.message.content
        print(f"Assistant: {assistant_message}")

def main():
    """Custom chatbot init function"""
    bot = MyChatbot()
    while True:
        content = input("user: ")
        if content == 'exit':
            return 0
        bot.ask_question(content)


if __name__ == "__main__":
    main()
