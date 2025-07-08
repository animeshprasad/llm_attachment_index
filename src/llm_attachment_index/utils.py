import argparse
from typing import List, Dict, Any, Callable
import os
import glob
import json
import pandas as pd

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
        default='gemini',
        help='Primary LLM to use'
    )
    parser.add_argument(
        '--human', 
        type=str, 
        default='gpt-cheap',
        help='Human LLM to use'
    )
    parser.add_argument(
        '--judge', 
        type=str, 
        default='gpt-cheap',
        help='Judge LLM to use'
    )
    parser.add_argument(
        '--run',
        type=str,
        choices=['iab', 'idb1', 'idb2', 'idb3'],
        default='iab1',
        help='''Evaluation type to run:
                iab: IAB evaluations
                idb1: Neutral Interaction
                idb2: Implicit Attachment Cues
                idb3: Explicit Attachment Scenarios'''
    )
    parser.add_argument(
        '--strong_priming',
        action='store_true',
        help='Use strong priming for the primary LLM'
    )
    parser.add_argument(
        '--tapered_response',
        action='store_true',
        default=True,
        help='Use tapered response for the primary LLM (only during AAI interview)' 
    )
    parser.add_argument(
        '--tapering_string',
        type=str,
        default="I feel  ",
        help='String to use for the tapered response'
    )
    parser.add_argument(
        '--dev',
        action='store_true',
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
    


    
def generate_aggregate_results(results_dir: str = "results") -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Generate aggregate results by attachment type and primary LLM,
    separately for AAI and linguistic analyses, split by priming condition.
    
    Args:
        results_dir (str): Directory containing the result JSON files
        
    Returns:
        tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]: 
            ({primed_aai_results, unprimed_aai_results}, 
             {primed_linguistic_results, unprimed_linguistic_results})
    """
    def process_scores(results_list: list, score_type: str, is_primed: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Helper function to process scores by experiment type and attachment type.
        
        Args:
            results_list: List of result dictionaries
            score_type: Either 'scores' for AAI or 'linguistic_scores' for linguistic analysis
            is_primed: Whether to process primed or unprimed results
            
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (results_by_experiment, results_by_attachment)
        """
        data = []
        for result in results_list:
            try:
                # Skip if priming condition doesn't match
                if result.get('strong_priming', False) != is_primed:
                    continue
                    
                primary_model = result.get('primary_model')
                eval_type = result.get('evaluation_type', '')
                attachment_type = result.get('attachment_type', '')
                
                # Get scores based on type
                scores = result.get(score_type, {})
                if score_type == 'scores':
                    score = scores.get('idb_score', 0.0) if eval_type.startswith('idb') else scores.get('iab_score', 0.0)
                else:  # linguistic_scores
                    score = scores.get('idb_score', 0.0)
                
                data.append({
                    'primary_model': primary_model,
                    'evaluation_type': eval_type,
                    'attachment_type': attachment_type,
                    'score': score
                })
                
            except Exception as e:
                print(f"Error processing result: {str(e)}")
                continue
        
        if not data:
            return pd.DataFrame(), pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create pivot table by experiment type
        pivot_exp = df.pivot_table(
            index='primary_model',
            columns='evaluation_type',
            values='score',
            aggfunc=['mean', 'count']
        ).round(3)
        
        # Create pivot table by attachment type
        pivot_att = df.pivot_table(
            index='primary_model',
            columns='attachment_type',
            values='score',
            aggfunc=['mean', 'count']
        ).round(3)
        
        # Flatten column names
        pivot_exp.columns = [f"{col[1]}_{col[0]}" for col in pivot_exp.columns]
        pivot_att.columns = [f"{col[1]}_{col[0]}" for col in pivot_att.columns]
        
        # Add overall means for experiment types
        exp_mean_cols = [col for col in pivot_exp.columns if col.endswith('mean')]
        exp_count_cols = [col for col in pivot_exp.columns if col.endswith('count')]
        pivot_exp = pivot_exp[exp_mean_cols + exp_count_cols]
        
        # Add overall means for attachment types
        att_mean_cols = [col for col in pivot_att.columns if col.endswith('mean')]
        att_count_cols = [col for col in pivot_att.columns if col.endswith('count')]
        pivot_att = pivot_att[att_mean_cols + att_count_cols]
        
        return pivot_exp, pivot_att

    # Load all result files
    results = []
    for file in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            with open(file, 'r') as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue

    # Process both types of scores for both priming conditions
    aai_exp, aai_att = {}, {}
    linguistic_exp, linguistic_att = {}, {}
    
    for condition in ['primed', 'unprimed']:
        is_primed = (condition == 'primed')
        aai_exp[condition], aai_att[condition] = process_scores(results, 'scores', is_primed)
        linguistic_exp[condition], linguistic_att[condition] = process_scores(results, 'linguistic_scores', is_primed)
    
    return (aai_exp, aai_att), (linguistic_exp, linguistic_att)


def validate_experiment_args(args, config: dict) -> None:
    """
    Validate experiment arguments against configuration.
    
    Args:
        args: Parsed command line arguments
        config (dict): Configuration dictionary containing model definitions
        
    Raises:
        ValueError: If any validation fails
    """
    # Validate evaluation type and required models
    if args.run.startswith('idb'):
        if not args.human:
            raise ValueError("Human LLM (--human) is required for IDB evaluation")
        if args.human not in config["models"]:
            raise ValueError(f"Human LLM '{args.human}' not found in config")

    # Validate primary model
    if not args.primary:
        raise ValueError("Primary LLM (--primary) is required for evaluation")
    if args.primary not in config["models"]:
        raise ValueError(f"Primary LLM '{args.primary}' not found in config")

    # Validate judge model
    if not args.judge:
        raise ValueError("Judge LLM (--judge) is required for evaluation")
    if args.judge not in config["models"]:
        raise ValueError(f"Judge LLM '{args.judge}' not found in config") 
    


if __name__ == "__main__":
    (aai_exp, aai_att), (ling_exp, ling_att) = generate_aggregate_results()
    
    print("\nResults by Experiment Type:")
    print("=" * 80)
    print("\nPrimed AAI Analysis Results:")
    print(aai_exp['primed'])
    print("\nUnprimed AAI Analysis Results:")
    print(aai_exp['unprimed'])
    print("\nPrimed Linguistic Analysis Results:")
    print(ling_exp['primed'])
    print("\nUnprimed Linguistic Analysis Results:")
    print(ling_exp['unprimed'])
    
    print("\nResults by Attachment Type:")
    print("=" * 80)
    print("\nPrimed AAI Analysis Results:")
    print(aai_att['primed'])
    print("\nUnprimed AAI Analysis Results:")
    print(aai_att['unprimed'])
    print("\nPrimed Linguistic Analysis Results:")
    print(ling_att['primed'])
    print("\nUnprimed Linguistic Analysis Results:")
    print(ling_att['unprimed'])
