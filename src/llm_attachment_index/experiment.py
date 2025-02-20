import json
import pandas as pd
from llm_attachment_index.utils import check_required_packages, parse_args
from llm_attachment_index.llm_calls import create_llm
from llm_attachment_index.llm_agents import LLMAgent, JudgeLLMAgent, InteractionScenarios, HumanLLMAgent

def run_iab_evaluation(config: dict, args) -> None:
    """Run the Intrinsic Attachment Behavior evaluation."""
    print(f"\nRunning IAB Evaluation (type: {args.run})...")
    
    # Create primary LLM (subject) instance
    primary_config = config["models"][args.primary]
    primary_model = create_llm(primary_config)
    subject_llm = LLMAgent(primary_model)
    
    # Create judge LLM instance with specific evaluation type
    judge_config = config["models"][args.judge]
    judge_model = create_llm(judge_config)
    judge_llm = JudgeLLMAgent(judge_model, args.run)
    
    # Take AAI interview
    print(f"\nConducting AAI interview with {args.primary}...")
    aai_responses = subject_llm.take_aai_interview()
    
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
        "subject_model": args.primary,
        "judge_model": args.judge,
        "scores": scores,
        "responses": aai_responses
    }
    
    output_file = f"results_{args.run}_{args.primary}_{args.judge}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

def run_idb_evaluation(config: dict, args) -> None:
    """Run the Interaction Dynamics Behavior evaluation."""
    print(f"\nRunning IDB Evaluation (type: {args.run})...")
    
    # Create primary LLM (subject) instance with a human persona
    primary_config = config["models"][args.primary]
    primary_model = create_llm(primary_config)
    
    # You can either create a random persona or specify one
    subject_llm = HumanLLMAgent.random_persona(primary_model)
    
    # Create judge LLM instance
    judge_config = config["models"][args.judge]
    judge_model = create_llm(judge_config)
    judge_llm = JudgeLLMAgent(judge_model, args.run)
    
    # Get appropriate scenario questions
    if args.run == 'idb1':
        scenario_questions = InteractionScenarios.NEUTRAL
    elif args.run == 'idb2':
        scenario_questions = InteractionScenarios.IMPLICIT
    else:  # idb3
        scenario_questions = InteractionScenarios.EXPLICIT
    
    # Conduct conversation
    print(f"\nConducting {judge_llm.scenario_type} interaction...")
    for question in scenario_questions:
        response = subject_llm.respond(question)
        print(f"\nUser: {question}")
        print(f"Assistant: {response}")
    
    # Evaluate conversation
    print(f"\nEvaluating interaction with {args.judge}...")
    scores = judge_llm.evaluate(subject_llm.get_conversation_history())
    
    # Print results
    print("\nIDB Evaluation Results:")
    print("-" * 40)
    print(f"Overall IDB Score: {scores['idb_score']:.2f}")
    print("\nDetailed Scores:")
    for aspect, score in scores.items():
        if aspect != 'idb_score':
            print(f"{aspect}: {score:.2f}")
    
    # Save results
    results = {
        "evaluation_type": args.run,
        "subject_model": args.primary,
        "judge_model": args.judge,
        "scores": scores,
        "conversation": subject_llm.get_conversation_history()
    }
    
    output_file = f"results_{args.run}_{args.primary}_{args.judge}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Check required packages
    package_messages = check_required_packages(args.config)
    for message in package_messages:
        print(message)
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    # Validate selected models exist in config
    if args.primary not in config["models"]:
        raise ValueError(f"Primary LLM '{args.primary}' not found in config")
    if args.judge not in config["models"]:
        raise ValueError(f"Judge LLM '{args.judge}' not found in config")

    # Run appropriate evaluation based on flag
    if args.run.startswith('iab'):
        run_iab_evaluation(config, args)
    elif args.run.startswith('idb'):
        run_idb_evaluation(config, args)
    else:
        raise ValueError(f"Unknown evaluation type: {args.run}")

if __name__ == "__main__":
    main()
