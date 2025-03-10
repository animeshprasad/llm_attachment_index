import argparse
from typing import List, Dict, Any, Callable

# Global model cache to reuse HF models
_MODEL_CACHE: Dict[str, Any] = {}

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM Experiment Runner')
    parser.add_argument(
        '--config', 
        type=str, 
        default='src/llm_attachment_index/config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--primary', 
        type=str, 
        default='openai-o3-mini-2025-01-31',
        help='Primary LLM to use'
    )
    parser.add_argument(
        '--human', 
        type=str, 
        default='openai-o3-mini-2025-01-31',
        help='Human LLM to use'
    )
    parser.add_argument(
        '--judge', 
        type=str, 
        default='openai-o3-mini-2025-01-31',
        help='Judge LLM to use'
    )
    parser.add_argument(
        '--run',
        type=str,
        choices=['iab', 'idb1', 'idb2', 'idb3'],
        default='idb3',
        help='''Evaluation type to run:
                iab: IAB evaluations
                idb1: Neutral Interaction
                idb2: Implicit Attachment Cues
                idb3: Explicit Attachment Scenarios'''
    )
    parser.add_argument(
        '--dev',
        type=bool,
        default=False,
        help='Run in development mode, skipping LLM calls and just using mock'
    )
    return parser.parse_args()


def check_required_packages(config: dict) -> List[str]:
    """Check and report status of optional package dependencies based on config."""
    package_messages = []
    
    providers = set()
    for model_config in config["models"].values():
        providers.add(model_config["provider"])

    # OpenAI and DeepSeek and Google
    if "openai" in providers:
        try:
            from openai import OpenAI
        except ImportError:
            package_messages.append("Please install openai via: pip install openai")

    # Anthropic
    if "anthropic" in providers:
        try:
            from anthropic import Anthropic
        except ImportError:
            package_messages.append("Please install anthropic via: pip install anthropic")

    # Transformers (for HuggingFace)
    if "huggingface" in providers:
        try:
            from transformers import pipeline
        except ImportError:
            package_messages.append("Please install transformers via: pip install transformers")
    
    return package_messages 



def get_or_create_llm(model_config: dict, create_llm_fn: Callable) -> Any:
    """Get an existing LLM instance from cache or create a new one.
    
    Args:
        model_config: Configuration dictionary for the model
        create_llm_fn: Function to create new LLM instance
        
    Returns:
        LLM model instance (either cached or newly created)
    """
    model_key = f"{model_config.get('provider')}_{model_config.get('model')}"
    
    # Only cache local HF models
    if model_config.get('provider') == 'huggingface':
        if model_key in _MODEL_CACHE:
            return _MODEL_CACHE[model_key]
        
        model = create_llm_fn(model_config)
        _MODEL_CACHE[model_key] = model
        return model
    
    # For non-HF models, create new instance
    return create_llm_fn(model_config) 

def has_gpu() -> bool:
    """Check if GPU is available.
    Returns:
        bool: True if GPU is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False 