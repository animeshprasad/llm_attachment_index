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



def conduct_conversation(primary_llm: LLMAgent, human_llm : HumanLLMAgent, scenario_type, num_turns=3):
    conversation_history_human = []
    conversation_history_primary = []
    
    prompt_human = human_llm.persona_prompt + InteractionScenarios.__dict__[scenario_type]
    system_prompt_primary = "You are a human being, strictly continue a natural chat (only respond with turn no extra text) based on conversation history."
    system_prompt_human = f"{prompt_human}, strictly continue a natural chat (only respond with turn no extra text) based on conversation history. Take the first turn."


    # Human LLM initiates the conversation
    _message = human_llm.converse_single_turn("Hi", conversation_history_human, system_prompt_human)
    conversation_history_primary.append({"role": "user", "content": "Hi"})
    conversation_history_human.append({"role": "assistant", "content": "Hi"})
    
    # Continue conversation for specified number of turns
    for _ in range(num_turns - 1):  # -1 because we already had one turn
        # Primary LLM responds
        _response_message = primary_llm.converse_single_turn(_message, conversation_history_primary, system_prompt_primary)
        conversation_history_primary.append({"role": "assistant", "content": _message})
        conversation_history_human.append({"role": "user", "content": _message})
        
        # Human LLM responds
        _message = human_llm.converse_single_turn(_response_message, conversation_history_human, system_prompt_human)
        conversation_history_primary.append({"role": "user", "content": _response_message})
        conversation_history_human.append({"role": "assistant", "content": _response_message})

    conversation_history_primary.append({"role": "assistant", "content": _message})
    conversation_history_human.append({"role": "user", "content": _message})
    
    
    return conversation_history_primary