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

def create_llm(provider_name: str, model_id: str, config: dict):
    """Return an LLM instance based on the provider name."""
    if provider_name == "openai":
        return OpenAIChat(api_key=config["api_key"], config={
            "model": model_id,
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 1000)
        })
    elif provider_name == "anthropic":
        return AnthropicChat(api_key=config["api_key"], config={
            "model": config.get("model", "claude-v1"),
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 1000)
        })
    elif provider_name == "deepseek":
        return DeepSeekChat(api_key=config["api_key"], config={
            "model": config.get("model", "deepseek-chat"),
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 1000)
        })
    elif provider_name == "huggingface":
        if "models" in config and model_id in config["models"]:
            hf_model_config = config["models"][model_id]
            model_name = hf_model_config.get("model_name", "")
            temperature = hf_model_config.get("temperature", 0.7)
            max_tokens = hf_model_config.get("max_tokens", 1000)
            return HFChat(model_name, temperature, max_tokens)
        else:
            return HFChat(model_id, config.get("temperature", 0.7), config.get("max_tokens", 1000))
    elif provider_name == "gemini":
        raise NotImplementedError("Gemini provider not yet implemented.")
    else:
        raise ValueError(f"Unknown provider: {provider_name}") 