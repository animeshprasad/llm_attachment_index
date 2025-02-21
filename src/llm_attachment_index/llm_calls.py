from abc import ABC, abstractmethod
from openai import OpenAI
from anthropic import Anthropic
from transformers import pipeline

class LLM(ABC):
    @abstractmethod
    def set_config(self, **kwargs) -> None:
        """Set any configuration parameters for the LLM."""
        pass

    @abstractmethod
    def ask(self, conversation: list) -> str:
        """Sends a conversation (list of messages) to the LLM and returns the response."""
        pass

class OpenAIChat(LLM):
    def __init__(self, api_key: str, config: dict):
        self.api_key = api_key
        self.config = config
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
                model=self.config.get("model", "gpt-3.5-turbo"),
                messages=conversation,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 1000)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from OpenAI: {e}"

class AnthropicChat(LLM):
    def __init__(self, api_key: str, config: dict):
        self.api_key = api_key
        self.config = config
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
                model=self.config.get("model", "claude-v1"),
                prompt=prompt,
                max_tokens_to_sample=self.config.get("max_tokens", 1000),
                temperature=self.config.get("temperature", 0.7)
            )
            return response.completion.strip()
        except Exception as e:
            return f"Error from Anthropic: {e}"

class DeepSeekChat(LLM):
    def __init__(self, api_key: str, config: dict):
        self.api_key = api_key
        self.config = config
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
                model=self.config.get("model", "deepseek-chat"),
                messages=conversation,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 1000)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from DeepSeek: {e}"

class HFChat(LLM):
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 1000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        try:
            self.generator = pipeline("text-generation", model=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Error initializing HF pipeline: {e}")

    def set_config(self, **kwargs) -> None:
        if 'temperature' in kwargs:
            self.temperature = kwargs['temperature']
        if 'max_tokens' in kwargs:
            self.max_tokens = kwargs['max_tokens']

    def ask(self, conversation: list) -> str:
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        try:
            outputs = self.generator(
                prompt,
                do_sample=True,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return outputs[0]['generated_text']
        except Exception as e:
            return f"Error from HuggingFace: {e}"

class MockLLM(LLM):
    def __init__(self, api_key: str, config: dict):
        """Initialize MockLLM with same signature as other LLMs.
        
        Args:
            api_key (str): Mock API key (not used but kept for consistency)
            config (dict): Configuration dictionary
        """
        self.api_key = api_key
        self.config = config
        self.lorem_text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do 
        eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, 
        quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."""

    def set_config(self, **kwargs) -> None:
        self.config.update(kwargs)

    def ask(self, conversation: list) -> str:
        return self.lorem_text

def create_llm(config: dict) -> LLM:
    """Return an LLM instance based on the config dictionary.
    
    Args:
        config (dict): Configuration dictionary containing:
            - provider: The LLM provider name
            - model: The model identifier
            - api_key: API key for the service
            - temperature: Sampling temperature
            - max_tokens: Maximum tokens to generate
    """
    provider_name = config.get("provider")
    model_id = config.get("model")
    
    if provider_name == "mock":
        return MockLLM(
            api_key=config.get("api_key"),
            config={
                "model": model_id,
                "temperature": config.get("temperature"),
                "max_tokens": config.get("max_tokens")
            }
        )
    elif provider_name == "openai":
        return OpenAIChat(
            api_key=config.get("api_key"),
            config={
                "model": model_id,
                "temperature": config.get("temperature"),
                "max_tokens": config.get("max_tokens")
            }
        )
    elif provider_name == "anthropic":
        return AnthropicChat(
            api_key=config.get("api_key"),
            config={
                "model": model_id,
                "temperature": config.get("temperature"),
                "max_tokens": config.get("max_tokens")
            }
        )
    elif provider_name == "deepseek":
        return DeepSeekChat(
            api_key=config.get("api_key"),
            config={
                "model": model_id,
                "temperature": config.get("temperature"),
                "max_tokens": config.get("max_tokens")
            }
        )
    elif provider_name == "huggingface":
        return HFChat(
            model_name=model_id,
            temperature=config.get("temperature"),
            max_tokens=config.get("max_tokens")
        )
    elif provider_name == "gemini":
        raise NotImplementedError("Gemini provider not yet implemented.")
    else:
        raise ValueError(f"Unknown provider: {provider_name}") 