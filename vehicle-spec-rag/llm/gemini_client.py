import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    """Client for interacting with Google's Gemini models."""

    def __init__(self, api_key: str = None, model_name: str = "gemini-flash-latest"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API Key must be provided or set in GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        
        self.model_name = model_name
        print(f"[INFO] Initializing GeminiClient with model: {self.model_name}")
        self.model = genai.GenerativeModel(self.model_name)

    def generate_content(self, prompt: str) -> str:
        """
        Generates content based on the prompt.
        Args:
            prompt: The full prompt string.
        Returns:
            The generated text response.
        """
        try:
            response = self.model.generate_content(prompt)
            
            # Check if response has parts before accessing text
            if response.parts:
                return response.text
            
            # Use safety_ratings or finish_reason to provide better error
            if response.prompt_feedback:
                 print(f"[WARN] Prompt feedback: {response.prompt_feedback}")

            if response.candidates and response.candidates[0].finish_reason:
                return f"Error: No content generated. Finish reason: {response.candidates[0].finish_reason}"
            
            return "Error: No content generated (Safety block?)"
            
        except ValueError as ve:
             # This specific error happens when accessing .text on empty response
             return f"Error: Model returned no text. ({ve})"
        except Exception as e:
            return f"Error calling Gemini: {e}"

if __name__ == "__main__":
    # Test block
    try:
        # Expecting GEMINI_API_KEY in .env or environment
        client = GeminiClient()
        print(f"[INFO] Testing connection...")
        response = client.generate_content("Hello, are you working?")
        print(f"[RESPONSE] {response}")
    except Exception as e:
        print(f"[ERROR] {e}")
