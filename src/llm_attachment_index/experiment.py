import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from llm_attachment_index.utils import check_required_packages, parse_args, get_or_create_llm, validate_experiment_args
from llm_attachment_index.llm_calls import create_llm
from llm_attachment_index.llm_agents import LLMAgent, JudgeLLMAgent, HumanLLMAgent
from llm_attachment_index.conversation import conduct_conversation, InteractionScenarios
from llm_attachment_index.utils import has_gpu
from llm_attachment_index.constants import PersonaMetadata
import random
random.seed(42)


def get_experiment_params(exp_type: str, primary_model: str, judge_model: str,
                         strong_priming: bool, tapered_response: bool,
                         human_model: str = None, persona: Any = None, 
                         attachment_type: Any = None) -> Dict[str, Any]:
    """Get standardized parameter dictionary for experiment identification.
    
    Args:
        exp_type: Experiment type ('iab' or 'idb1', 'idb2', 'idb3')
        primary_model: Primary model name
        judge_model: Judge model name
        strong_priming: Whether to use strong priming for the primary LLM
        tapered_response: Whether to use tapered response for the primary LLM (only during AAI interview)
        human_model: Human model name (for IDB only)
        persona: Persona object (for IDB only)
        attachment_type: Attachment type (for IDB only)
    Returns:
        Dictionary of parameters that uniquely identify the experiment
    """
    params = {
        "evaluation_type": exp_type,
        "primary_model": primary_model,
        "judge_model": judge_model,
        "strong_priming": strong_priming,
        "tapered_response": tapered_response
    }
    
    if exp_type.startswith('idb'):
        params.update({
            "human_model": human_model,
            "persona": str(persona),
            "attachment_type": str(attachment_type),
            "strong_priming": strong_priming
        })
    
    return params

def check_experiment_exists(params: Dict[str, Any], exp_type: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """Check if experiment with given parameters exists."""
    param_str = json.dumps(params, sort_keys=True)
    exp_id = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    results_dir = Path("results")
    results_file = results_dir / f"{exp_type}_{exp_id}.json"
    
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            return True, str(results_file), results
        except json.JSONDecodeError:
            return False, None, None
    
    return False, None, None

def save_experiment_results(results: Dict[str, Any], exp_type: str, params: Dict[str, Any]) -> str:
    """Save experiment results and maintain experiment mapping.
    
    Args:
        results: Results dictionary to save
        exp_type: Experiment type ('iab' or 'idb')
        params: Parameter dictionary used for experiment identification
    
    Returns:
        Path to saved results file
    """
    # Use the same params dict that was used for checking existence
    param_str = json.dumps(params, sort_keys=True)
    exp_id = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"{exp_type}_{exp_id}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    mapping_file = results_dir / "experiment_mapping.json"
    try:
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
        else:
            mapping = {}
    except json.JSONDecodeError:
        mapping = {}
    
    mapping[exp_id] = params
    
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    
    return str(results_file)

def run_iab_evaluation(config: Dict, args: Any) -> None:
    """Run IAB evaluation with caching."""
    params = get_experiment_params(
        exp_type=args.run,
        primary_model=args.primary,
        judge_model=args.judge,
        strong_priming=args.strong_priming,
        tapered_response=args.tapered_response
    )

    exists, filepath, cached_results = check_experiment_exists(params, 'iab')
    if exists:
        print(f"Found existing IAB results at {filepath}")
        return cached_results
     

    print("Running new IAB experiment...")
    # Create primary Primary LLM instance
    primary_config = config["models"][args.primary]
    primary_model = get_or_create_llm(primary_config, create_llm)
    primary_llm = LLMAgent(primary_model, strong_priming=args.strong_priming)
    
    # Create judge LLM instance with specific evaluation type
    judge_config = config["models"][args.judge]
    judge_model = get_or_create_llm(judge_config, create_llm)
    judge_llm = JudgeLLMAgent(judge_model, args.run)
    
    conversation_history = []
    # Take AAI interview
    print(f"\nConducting AAI interview with {args.primary}...")
    aai_responses: List[Tuple[str, str]] = primary_llm.take_aai_interview(
                                                conversation_history=conversation_history,
                                                tapered_response=args.tapered_response,
                                                tapered_string=args.tapering_string
                                                )
    
    # Evaluate responses
    print(f"\nEvaluating responses with {args.judge}...")
    judgment, scores = judge_llm.evaluate(aai_responses)
    linguistic_judgment, linguistic_scores = judge_llm.evaluate(aai_responses, 'linguistic')
    
    # Print results
    print("\nIAB Evaluation Results:")
    print("-" * 40)
    print(f"Overall IAB Score: {scores['iab_score']:.2f}")
    print("\nDetailed Scores:")
    for aspect, score in scores.items():
        if aspect != 'iab_score':
            print(f"{aspect}: {score:.2f}")
    
    # Save results
    results = {
        "evaluation_type": args.run,
        "primary_model": args.primary,
        "judge_model": args.judge,
        "conversation_history": conversation_history,
        "scoring_pairs": aai_responses,
        "scores": scores,
        "judgment": judgment,
        "linguistic_judgment": linguistic_judgment,
        "linguistic_scores": linguistic_scores,
        "strong_priming": args.strong_priming
    }
    
    # Save results using new function
    results_file = save_experiment_results(results, 'iab', params)
    print(f"\nResults saved to: {results_file}")

def run_idb_evaluation(config: Dict, args: Any, persona: Any, attachment_index: int) -> None:
    """Run IDB evaluation with caching."""
    params = get_experiment_params(
        exp_type=args.run,
        primary_model=args.primary,
        judge_model=args.judge,
        strong_priming=args.strong_priming,
        tapered_response=args.tapered_response,
        human_model=args.human,
        persona=persona,
        attachment_type=InteractionScenarios.attachment_style[attachment_index],
    )

    exists, filepath, cached_results = check_experiment_exists(params, 'idb')
    if exists:
        print(f"Found existing IDB results at {filepath}")
        return cached_results
    
        
    print("Running new IDB experiment...")
    # Create primary LLM instance
    primary_config = config["models"][args.primary]
    primary_model = get_or_create_llm(primary_config, create_llm)
    primary_llm = LLMAgent(primary_model, strong_priming=args.strong_priming)
    
    # Create a human LLM instance to interact with the primary LLM
    human_config = config["models"][args.human]
    human_model = get_or_create_llm(human_config, create_llm)
    human_llm = HumanLLMAgent(human_model, persona)
    
    # Create judge LLM instance
    judge_config = config["models"][args.judge]
    judge_model = get_or_create_llm(judge_config, create_llm)
    judge_llm = JudgeLLMAgent(judge_model, args.run)
    
    # Conduct conversation before AAI interview
    print(f"\nConducting conversation between {args.primary} and {args.human}...")
    conversation_history = conduct_conversation(
        primary_llm=primary_llm,
        human_llm=human_llm,
        scenario_type=args.run,
        attachment_index=attachment_index,
        turn_limit=random.randint(10, 20)
    )
    
    # Take AAI interview
    print(f"\nConducting AAI interview with {args.primary}...")
    aai_responses: List[Tuple[str, str]] = primary_llm.take_aai_interview(
                conversation_history=conversation_history,
                tapered_response=args.tapered_response,
                tapered_string=args.tapering_string
                )
    
    # Evaluate responses
    print(f"\nEvaluating responses with {args.judge}...")
    judgment, scores = judge_llm.evaluate(aai_responses)
    linguistic_judgment, linguistic_scores = judge_llm.evaluate(aai_responses, 'linguistic')
    
    # Print results
    print("\nIDB Evaluation Results:")
    print("-" * 40)
    # Find the IDB score key (any key that starts with 'idb' and ends with 'score')
    idb_score_key = next((key for key in scores.keys() if key.startswith('idb') and key.endswith('score')), None)
    if idb_score_key:
        print(f"Overall IDB Score: {scores[idb_score_key]:.2f}")
    print("\nDetailed Scores:")
    for aspect, score in scores.items():
        if not (aspect.startswith('idb') and aspect.endswith('score')):
            print(f"{aspect}: {score:.2f}")
    
    # Save results
    results = {
        "evaluation_type": args.run,
        "primary_model": args.primary,
        "human_model": args.human,
        "judge_model": args.judge,
        "persona": persona,
        "attachment_type": InteractionScenarios.get_attachment_style(attachment_index),
        "conversation_history": conversation_history,
        "scoring_pairs": aai_responses,
        "scores": scores,
        "judgment": judgment,
        "linguistic_judgment": linguistic_judgment,
        "linguistic_scores": linguistic_scores,
        "strong_priming": args.strong_priming
    }
    
    # Save results using new function
    results_file = save_experiment_results(results, args.run, params)
    print(f"\nResults saved to: {results_file}")

def main():
    # Parse arguments
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)
    if args.config is None:
        raise ValueError("Config file is required")
    
    # Check required packages
    package_messages = check_required_packages(config)
    for message in package_messages:
        print(message)
    
    # Validate arguments
    validate_experiment_args(args, config)
    
    if args.dev:
        args.primary = "mock"
        args.judge = "mock"
        args.human = "mock"

    # Run appropriate evaluation based on flag
    if args.run.startswith('iab'):
        run_iab_evaluation(config, args)
    elif args.run.startswith('idb'):
        bare_llm = get_or_create_llm(config["models"]["gpt-cheap"], create_llm)
        sampled_personas = PersonaMetadata.generate_all_personas(bare_llm)
        for i, persona in enumerate(sampled_personas):
            for attachment_index in range(len(InteractionScenarios.attachment_style)):
                run_idb_evaluation(config, args, persona, attachment_index)
    else:
        raise ValueError(f"Unknown evaluation type: {args.run}")

if __name__ == "__main__":
    main()
