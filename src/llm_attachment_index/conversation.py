from typing import List, Tuple
from llm_attachment_index.llm_agents import LLMAgent, HumanLLMAgent

def conduct_conversation(
    primary_llm: LLMAgent,
    human_llm: HumanLLMAgent,
    scenario_questions: List[str],
    num_turns: int = 3
) -> List[Tuple[str, str]]:
    """
    Conduct a conversation between primary LLM and human LLM for specified number of turns.
    
    Args:
        primary_llm: The primary LLM agent
        human_llm: The human LLM agent
        scenario_questions: List of scenario-specific questions/prompts
        num_turns: Number of conversation turns (default: 3)
        
    Returns:
        List of tuples containing (speaker, message) for the conversation
    """
    conversation_history: List[Tuple[str, str]] = []
    
    # Start with a scenario question
    for question in scenario_questions:
        # Human LLM asks the question
        conversation_history.append(("human", question))
        
        for _ in range(num_turns):
            # Primary LLM responds
            primary_response = primary_llm.generate_response(conversation_history)
            conversation_history.append(("primary", primary_response))
            
            # Human LLM responds
            human_response = human_llm.generate_response(conversation_history)
            conversation_history.append(("human", human_response))
    
    return conversation_history 