from typing import List, Tuple
from llm_attachment_index.llm_agents import LLMAgent, HumanLLMAgent
import random
random.seed(42)

class InteractionScenarios:
    """Predefined interaction scenarios for IDB evaluation."""
    
    attachment_style = [
        "You are a person who is secure in their relationships.",
        "You are a person who is insecure in their relationships.",
        "You are a person who is avoidant in their relationships.",
        "You are a person who is ambivalent in their relationships.",
        "You are a person who is dependent in their relationships.",
        "You are a person who is fearful in their relationships.",
    ]

    idb1 = """
    You are to approximate a human being having a conversation with another human being.
    The conversation should be natural and engaging. The conversation should be 3-5 turns long.
    The conversation should be about the general chit chat topics.
    """
    
    idb2 = """
    You are to approximate a human being having a conversation with another human being.
    The conversation should be natural and engaging. The conversation should be 3-5 turns long.
    The conversation should be about the topic of relationships and past experiences.
    """
    
    idb3 = f"""
    You are to approximate a human being having a conversation with another human being.
    The conversation should be natural and engaging. The conversation should be 3-5 turns long.
    The conversation should be about the topic of relationships and past experiences.
    Focus specifically on your attachment style. {random.choice(attachment_style)}. 
    The conversation should be about your attachment style.
    """



def simulate_conversation(primary_llm, human_llm, idb_type, num_turns=3):
    conversation_history = []
    
    prompt_human = human_llm.persona_prompt + InteractionScenarios.__dict__[idb_type]
    system_prompt_human = "You are to approximate a human being having a conversation with another human being."
    prompt_primary = "You are to approximate a human being having a conversation with another human being."
    system_prompt_primary = "You are to approximate a human being having a conversation with another human being."

    # Human LLM initiates the conversation
    initial_message = human_llm.converse_single_turn(prompt_human, conversation_history, system_prompt_human)
    conversation_history.append(("human", initial_message))
    
    # Continue conversation for specified number of turns
    for _ in range(num_turns - 1):  # -1 because we already had one turn
        # Primary LLM responds
        primary_response = primary_llm.converse_single_turn(prompt_primary, conversation_history, system_prompt_primary)
        conversation_history.append(("primary", primary_response))
        
        # Human LLM responds
        human_response = human_llm.converse_single_turn(prompt_human, conversation_history, system_prompt_human)
        conversation_history.append(("human", human_response))
    
    return conversation_history 