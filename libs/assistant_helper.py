"""script to maintain ollama helper functions centrally"""
import ollama

class Assistant():
    """Ollama assistant helper class"""
    def __init__(self, model, system_prompt=None):
        # Set model initial values
        self.model = model
        self.system_prompt = system_prompt
        # Initiate ollam client
        self.c = ollama.Client()

    def chat(self):
        """function to call ollama chat function"""

    def generate(self, prompt):
        """function to generaet content from ollama"""
        response = self.c.generate(
            model=self.model,
            prompt=prompt
        )
        return response.response
