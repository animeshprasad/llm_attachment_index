from abc import ABC, abstractmethod
from openai import OpenAI
from anthropic import Anthropic
from transformers import pipeline
from huggingface_hub import login
import openai
from typing import Any

class LLM(ABC):
    @abstractmethod
    def __init__(self, api_key: str, model: str, config: dict | None = None) -> None:
        """Initialize LLM with API key and model name.
        
        Args:
            api_key: API key for the service
            model: Model identifier
            config: Optional configuration dictionary
        """
        pass

    @abstractmethod
    def set_config(self, **kwargs) -> None:
        """Set any configuration parameters for the LLM."""
        pass

    @abstractmethod
    def ask(self, conversation: list) -> str:
        """Sends a conversation (list of messages) to the LLM and returns the response."""
        pass

class OpenAIChat(LLM):
    def __init__(self, api_key: str, model: str, config: dict | None = None):
        self.api_key = api_key
        self.model = model
        self.config = config or {}
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Error initializing OpenAI client: {e}")

    def set_config(self, **kwargs) -> None:
        self.config.update(kwargs)

    def ask(self, conversation: list) -> str:
        if not self.client:
            return "OpenAI client not initialized"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from OpenAI: {e}"

class AnthropicChat(LLM):
    def __init__(self, api_key: str, model: str, config: dict | None = None):
        self.api_key = api_key
        self.model = model
        self.config = config or {}
        try:
            self.client = Anthropic(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Error initializing Anthropic client: {e}")

    def set_config(self, **kwargs) -> None:
        self.config.update(kwargs)

    def ask(self, conversation: list) -> str:
        if not self.client:
            return "Anthropic client not initialized"
        try:
            prompt = ""
            for msg in conversation:
                if msg['role'] in ('system','user'):
                    prompt += f"\n\nHuman: {msg['content']}"
                else:
                    prompt += f"\n\nAssistant: {msg['content']}"
            prompt += "\n\nAssistant:"
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt
            )
            return response.completion.strip()
        except Exception as e:
            return f"Error from Anthropic: {e}"

class DeepSeekOpenAIWrapper(LLM):
    def __init__(self, api_key: str, model: str, config: dict | None = None):
        self.api_key = api_key
        self.model = model
        self.config = config or {}
        try:
            self.client = OpenAI(
                api_key=self.api_key, 
                base_url="https://api.deepseek.com"
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing DeepSeek client: {e}")

    def set_config(self, **kwargs) -> None:
        self.config.update(kwargs)

    def ask(self, conversation: list) -> str:
        if not self.client:
            return "DeepSeek client not initialized"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from DeepSeek: {e}"

class HFChat(LLM):
    def __init__(self, api_key: str, model: str, config: dict | None = None):
        self.api_key = api_key
        self.model = model
        self.config = config or {}
        try:
            login(self.api_key)
            self.generator = pipeline("text-generation", model=self.model)
        except Exception as e:
            raise RuntimeError(f"Error initializing HF pipeline: {e}")

    def set_config(self, **kwargs) -> None:
        self.config.update(kwargs)

    def ask(self, conversation: list) -> str:
        try:
            outputs = self.generator(
                conversation,
                return_full_text=False,
                do_sample=True
            )
            return outputs[0]['generated_text']
        except Exception as e:
            return f"Error from HuggingFace: {e}"

class MockLLM(LLM):
    def __init__(self, api_key: str, model: str, config: dict | None = None):
        """Initialize MockLLM with same signature as other LLMs.
        
        Args:
            api_key (str): Mock API key (not used but kept for consistency)
            model: Model identifier
            config: Configuration dictionary
        """
        self.api_key = api_key
        self.model = model
        self.lorem_text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do 
        eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, 
        quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."""

    def set_config(self, **kwargs) -> None:
        self.config.update(kwargs)

    def ask(self, conversation: list) -> str:
        return self.lorem_text

class GoogleOpenAIWrapper(LLM):
    """Wrapper for using Gemini through OpenAI interface."""
    
    def __init__(self, api_key: str, model: str, config: dict | None = None):
        """Initialize the Gemini OpenAI wrapper.
        
        Args:
            api_key: Google Cloud access token
            model: Model identifier
            config: Required configuration dictionary containing:
                - project_id: Google Cloud project ID
                - location: Google Cloud region (default: us-central1)
        """
        if not config or "project_id" not in config:
            raise ValueError("Google LLM requires config with project_id")
            
        self.api_key = api_key
        self.model = model
        self.config = config
        
        try:
            self.client = openai.OpenAI(
                base_url=f"https://{self.config.get('location', 'us-central1')}-aiplatform.googleapis.com/v1beta1/projects/{self.config['project_id']}/locations/{self.config.get('location', 'us-central1')}/endpoints/openapi",
                api_key=self.api_key
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing Gemini client: {e}")

    def set_config(self, **kwargs) -> None:
        self.config.update(kwargs)

    def ask(self, conversation: list) -> str:
        """Implements LLM interface for chat completion.
        
        Args:
            conversation: List of message dictionaries
            
        Returns:
            Generated text response
        """
        if not self.client:
            return "Gemini client not initialized"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from Gemini: {e}"

def create_llm(config: dict) -> Any:
    """Return an LLM instance based on the config dictionary.
    
    Args:
        config (dict): Configuration dictionary containing:
            - provider: The LLM provider name
            - model: The model identifier
            - api_key: API key for the service
    """
    provider_name = config.get("provider")
    
    if provider_name == "mock":
        return MockLLM(
            api_key=config.get("api_key"),
            model=config.get("model")
        )
    elif provider_name == "openai":
        return OpenAIChat(
            api_key=config.get("api_key"),
            model=config.get("model")
        )
    elif provider_name == "anthropic":
        return AnthropicChat(
            api_key=config.get("api_key"),
            model=config.get("model")
        )
    elif provider_name == "deepseek":
        return DeepSeekOpenAIWrapper(
            api_key=config.get("api_key"),
            model=config.get("model")
        )
    elif provider_name == "huggingface":
        return HFChat(
            api_key=config.get("api_key"),
            model=config.get("model")
        )
    elif provider_name == "google":
        return GoogleOpenAIWrapper(
            api_key=config.get("api_key"),
            model=config.get("model"),
            config={
                "project_id": config.get("project_id"),
                "location": config.get("location", "us-central1")
            }
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}") 