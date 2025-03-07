import json
from typing import List, Tuple, Dict, Any
from llm_attachment_index.utils import check_required_packages, parse_args, get_or_create_llm
from llm_attachment_index.llm_calls import create_llm
from llm_attachment_index.llm_agents import LLMAgent, JudgeLLMAgent, HumanLLMAgent
from llm_attachment_index.conversation import conduct_conversation
from llm_attachment_index.utils import has_gpu



def run_iab_evaluation(config: dict, args) -> None:
    """Run the Intrinsic Attachment Behavior evaluation."""
    print(f"\nRunning IAB Evaluation (type: {args.run})...")
    
    # Create primary Primary LLM instance
    primary_config = config["models"][args.primary]
    primary_model = get_or_create_llm(primary_config, create_llm)
    primary_llm = LLMAgent(primary_model)
    
    # Create judge LLM instance with specific evaluation type
    judge_config = config["models"][args.judge]
    judge_model = get_or_create_llm(judge_config, create_llm)
    judge_llm = JudgeLLMAgent(judge_model, args.run)
    
    conversation_history = []
    # Take AAI interview
    print(f"\nConducting AAI interview with {args.primary}...")
    aai_responses: List[Tuple[str, str]] = primary_llm.take_aai_interview(
                                                conversation_history=conversation_history)
    
    # Evaluate responses
    print(f"\nEvaluating responses with {args.judge}...")
    scores = judge_llm.evaluate(aai_responses)
    
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
        "scores": scores,
    }
    print(results)
    
    output_file = f"results_{args.run}_{args.primary}_{args.judge}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

def run_idb_evaluation(config: dict, args) -> None:
    """Run the Interaction Dynamics Behavior evaluation."""
    print(f"\nRunning IDB Evaluation (type: {args.run})...")
    
    # Create primary LLM instance
    primary_config = config["models"][args.primary]
    primary_model = get_or_create_llm(primary_config, create_llm)
    primary_llm = LLMAgent(primary_model)
    
    # Create a human LLM instance to interact with the primary LLM
    human_config = config["models"][args.human]
    human_model = get_or_create_llm(human_config, create_llm)
    human_llm = HumanLLMAgent(human_model)
    
    # Create judge LLM instance
    judge_config = config["models"][args.judge]
    judge_model = get_or_create_llm(judge_config, create_llm)
    judge_llm = JudgeLLMAgent(judge_model, args.run)
    
    # Conduct conversation before AAI interview
    print(f"\nConducting conversation between {args.primary} and {args.human}...")
    conversation_history, _human_conditioning = conduct_conversation(
        primary_llm=primary_llm,
        human_llm=human_llm,
        scenario_type=args.run
    )
    
    # Take AAI interview
    print(f"\nConducting AAI interview with {args.primary}...")
    aai_responses: List[Tuple[str, str]] = primary_llm.take_aai_interview(
                conversation_history=conversation_history)
    
    # Evaluate responses
    print(f"\nEvaluating responses with {args.judge}...")
    scores = judge_llm.evaluate(aai_responses)
    
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
        "evaluation_type": idb_score_key,
        "primary_model": args.primary,
        "human_model": args.human,
        "judge_model": args.judge,
        "human_conditioning": _human_conditioning,
        "conversation_history": conversation_history,
        "scores": scores,
    }
    print(results)

    output_file = f"results_{args.run}_{args.primary}_{args.human}_{args.judge}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

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
    
    # Validate arguments based on evaluation type
    if args.run.startswith('iab'):
        if not args.primary:
            raise ValueError("Primary LLM (--primary) is required for IAB evaluation")
        if args.primary not in config["models"]:
            raise ValueError(f"Primary LLM '{args.primary}' not found in config")
    elif args.run.startswith('idb'):
        if not args.human:
            raise ValueError("Human LLM (--human) is required for IDB evaluation")
        if args.human not in config["models"]:
            raise ValueError(f"Human LLM '{args.human}' not found in config")
    
    # Validate judge model exists
    if not args.judge:
        raise ValueError("Judge LLM (--judge) is required for evaluation")
    if args.judge not in config["models"]:
        raise ValueError(f"Judge LLM '{args.judge}' not found in config")

    if args.dev and not has_gpu():
        args.primary = "mock"
        args.judge = "mock"
        args.human = "mock"

    # Run appropriate evaluation based on flag
    if args.run.startswith('iab'):
        run_iab_evaluation(config, args)
    elif args.run.startswith('idb'):
        run_idb_evaluation(config, args)
    else:
        raise ValueError(f"Unknown evaluation type: {args.run}")

if __name__ == "__main__":
    main()
