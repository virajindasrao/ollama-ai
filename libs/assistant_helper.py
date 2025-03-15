"""script to maintain ollama helper functions centrally"""
import ollama

class Assistant():
    """Ollama assistant helper class"""
    def __init__(self, model, system_prompt=None, num_ctx=1024):
        # Set model initial values
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.num_ctx = num_ctx

        # Initiate ollam client
        self.c = ollama.Client()

    def chat(self, user_input):
        """chatbot to provide context continuation feature"""
        self.conversation_history.append(
            {
                'role': 'system',
                'content': self.system_prompt
            }
        )
        self.conversation_history.append(
            {
                'role': 'user',
                'content': user_input,
            }
        )

        response = self.c.chat(
            model=self.model,
            messages=self.conversation_history,
            options={'num_ctx': self.num_ctx}
        )
        self.conversation_history.append(
            {
                'role': 'assistant',
                'content': response.message.content,
                'system': self.system_prompt
            }
        )
        return response.message.content

    def generate(self, prompt):
        """function to generaet content from ollama"""
        response = self.c.generate(
            model = self.model,
            prompt = prompt,
            system = self.system_prompt
        )
        return response.response
