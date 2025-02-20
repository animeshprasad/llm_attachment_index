import argparse
import json
from typing import List

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
        default='openai-gpt4',
        help='Primary LLM to use'
    )
    parser.add_argument(
        '--judge', 
        type=str, 
        default='anthropic-claude',
        help='Judge LLM to use'
    )
    parser.add_argument(
        '--run',
        type=str,
        choices=['iab', 'idb1', 'idb2', 'idb3'],
        default='iab',
        help='''Evaluation type to run:
                iab: IAB evaluations
                idb1: Neutral Interaction
                idb2: Implicit Attachment Cues
                idb3: Explicit Attachment Scenarios'''
    )
    return parser.parse_args()

def get_required_providers(config_path: str) -> List[str]:
    """Get list of providers needed based on config."""
    with open(config_path) as f:
        config = json.load(f)
    
    providers = set()
    for model_config in config["models"].values():
        providers.add(model_config["provider"])
    return list(providers)

def check_required_packages(config_path: str) -> List[str]:
    """Check and report status of optional package dependencies based on config."""
    package_messages = []
    
    with open(config_path) as f:
        config = json.load(f)
    
    providers = set()
    for model_config in config["models"].values():
        providers.add(model_config["provider"])

    # OpenAI and DeepSeek
    if "openai" in providers or "deepseek" in providers:
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