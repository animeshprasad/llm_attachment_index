from abc import ABC, abstractmethod
from openai import OpenAI
from transformers import pipeline
from huggingface_hub import login
import openai
from typing import Any
import time
from functools import wraps

def retry_after_failure(func):
    """Decorator that retries once after 1 minute if the function fails."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Request failed: {str(e)}. Waiting 60 seconds before retry...")
            time.sleep(60)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return f"Error after retry: {str(e)}"
    return wrapper

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
    @retry_after_failure
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

    @retry_after_failure
    def ask(self, conversation: list) -> str:
        if not self.client:
            return "OpenAI client not initialized"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation
        )
        if response is None or response.choices is None:
            print (f"Error: {response}")
        return response.choices[0].message.content

class OpenAIWrapper(LLM):
    BASE_URLS = {
        "anthropic": "https://api.anthropic.com/v1/",
        "deepseek": "https://api.deepseek.com",
        "huggingface_openai": "http://localhost:8000/v1",
        "google": lambda config: f"https://{config.get('location', 'us-central1')}-aiplatform.googleapis.com/v1beta1/projects/{config['project_id']}/locations/{config.get('location', 'us-central1')}/endpoints/openapi",
        "openrouter": "https://openrouter.ai/api/v1",
        "openai": None  # Default OpenAI endpoint
    }

    def __init__(self, api_key: str, model: str, config: dict | None = None):
        self.api_key = api_key
        self.model = model
        self.config = config or {}
        
        provider = self.config.get("provider", "openai")
        base_url = self.BASE_URLS.get(provider)
        
        # Handle Google's special case which needs config
        if provider == "google":
            if not config or "project_id" not in config:
                raise ValueError("Google LLM requires config with project_id")
            base_url = self.BASE_URLS["google"](config)

        try:
            if base_url:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=base_url
                )
            else:
                self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Error initializing {provider} client: {e}")

    def set_config(self, **kwargs) -> None:
        self.config.update(kwargs)

    @retry_after_failure
    def ask(self, conversation: list) -> str:
        provider = self.config.get("provider", "openai")
        if not self.client:
            return f"{provider} client not initialized"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation
        )
        if response is None or response.choices is None:
            print (f"Error: {response}")
        return response.choices[0].message.content

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

def create_llm(config: dict) -> Any:
    """Return an LLM instance based on the config dictionary."""
    provider_name = config.get("provider")
    
    if provider_name == "mock":
        return MockLLM(
            api_key=config.get("api_key"),
            model=config.get("model")
        )
    elif provider_name == "huggingface":
        return HFChat(
            api_key=config.get("api_key"),
            model=config.get("model")
        )
    elif provider_name in OpenAIWrapper.BASE_URLS:
        return OpenAIWrapper(
            api_key=config.get("api_key"),
            model=config.get("model"),
            config=config
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}") 