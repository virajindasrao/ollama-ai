"""script to demonstrate ollama chatbox with system prompt and conversation context"""
from libs.assistant_helper import Assistant


# Function to interact with the chatbot
class MyChatbot():
    """Custom chatbot class"""
    def __init__(self):
        self.system_prompt = 'You are a helpful assistant in cloud devops. Answer questions about cloud devops, provide advice, and assist with troubleshooting problems as best you can.'
        # The given model should be downloaded on local before calling from the script
        # To download the model run command `ollama pull llama3.2`
        # To confirm the download run command `ollama show llama3.2`
        # To download ollama on local check this and follow the steps : https://ollama.com/download
        self.assistant = Assistant('llama3.2', self.system_prompt)

    def set_system_prompt(self, system_prompt):
        """Function to set the system prompt"""
        self.system_prompt = system_prompt

    def ask_question(self, user_input):
        """Custom chatbot custom initiator"""
        # Append user's question to the conversation history
        assistant_response = self.assistant.chat(user_input)
        # Extract the assistant's message
        print(f" assistant: {assistant_response}")
        print("******************************************************")


# Example interaction
# ask_question('What is continuous integration?')
# ask_question('How does it differ from continuous deployment?')
def main():
    """Custom chatbot init function"""
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
