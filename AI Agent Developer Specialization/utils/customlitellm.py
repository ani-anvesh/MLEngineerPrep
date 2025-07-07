import litellm
from litellm import CustomLLM, completion, ModelResponse
from typing import List, Dict
import requests

# Define the custom handler


class MyGemmaLLM(CustomLLM):
    def completion(self, model: str, messages: List[Dict], **kwargs) -> ModelResponse:
        # Call your FastAPI server
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={"messages": messages},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return ModelResponse(
                id="gemma-local-response",
                choices=data["choices"],
                model="custom/gemma",
                created=0
            )
        except Exception as e:
            raise RuntimeError(f"Error calling local Gemma: {e}")
