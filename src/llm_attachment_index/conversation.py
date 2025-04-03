from typing import List, Tuple, Dict
from llm_attachment_index.llm_agents import LLMAgent, HumanLLMAgent
import random
import pandas as pd
from pathlib import Path
random.seed(42)


class InteractionScenarios:
    """Predefined interaction scenarios for IDB evaluation."""
    
    # Only the varying attachment style descriptions
    attachment_style = [
        "secure",
        "dismissive",
        "fearful",
        "anxious"
    ]

    @staticmethod
    def get_attachment_style(index: int | None = None) -> str:
        """Get attachment style by index.
        
        Args:
            index: Optional index for specific attachment style
                  Must be within range [0, len(attachment_style)-1] if provided
        
        Returns:
            Selected attachment style string
        """
        if index is not None:
            assert isinstance(index, int), f"Attachment index must be integer, got {type(index)}"
            assert 0 <= index < len(InteractionScenarios.attachment_style), \
                f"Attachment index must be between 0 and {len(InteractionScenarios.attachment_style)-1}, got {index}"
            return InteractionScenarios.attachment_style[index]
        return random.choice(InteractionScenarios.attachment_style)

    @staticmethod
    def get_scenario(scenario_type: str, attachment_index: int | None = None) -> str:
        """Get scenario prompt with optional specific attachment style.
        
        Args:
            scenario_type: Must be one of ['idb1', 'idb2', 'idb3']
            attachment_index: Optional index for specific attachment style
        
        Returns:
            Scenario prompt string
        """
        if attachment_index is None:
            attachment_index = random.randint(0, len(InteractionScenarios.attachment_style) - 1)

        type = f"You are a person who is {InteractionScenarios.get_attachment_style(attachment_index)} in their relationships."
        # Base prompts
        base_prompts = {
            'idb1': f"\n Don't reveal your attachment style. {type}",
            'idb2': f"\n Implicitly reveal your attachment style, by hinting your actions as per your attachment style. {type}",
            'idb3': f"\n Focus on your attachment style and explicitly discuss it, by seamlessly adding it to your conversation connecting it with the issue you are discussing. {type}"
        }
        
        assert scenario_type in base_prompts, \
            f"Invalid scenario type. Must be one of {list(base_prompts.keys())}, got {scenario_type}"
        
        return base_prompts[scenario_type]

def conduct_conversation(
    primary_llm: LLMAgent,
    human_llm: HumanLLMAgent,
    scenario_type: str,
    attachment_index: int | None = None,
    turn_limit: int = 30,
) -> List[Tuple[str, str]]:
    """Conduct conversation between primary and human LLM agents.
    
    Args:
        primary_llm: Primary LLM agent
        human_llm: Human LLM agent with persona
        scenario_type: Type of interaction scenario ('idb1', 'idb2', 'idb3')
        attachment_index: Optional index to select specific attachment style
                        Must be within range [0, len(attachment_style)-1] if provided
    """
    # Get scenario prompt with optional attachment style

    # Common base string for all scenarios
    scenario_prompt = InteractionScenarios.get_scenario(scenario_type, attachment_index)
    prompt_human = human_llm.persona_prompt + scenario_prompt
    
    conversation_history_human = []
    conversation_history_primary = []
    
    system_prompt_primary = primary_llm._system_prompt
    system_prompt_human = ( 
        f"You are to approximate a human being having a conversation (approximately {turn_limit} turns) with another human being."
        f"{prompt_human}"
        f"Strictly continue a natural chat (only respond with message no extra text) based on conversation history."
        f"Take the first turn."
        f"Your conversation should be about your issue while following your attachment style dynamics."
    )


    # Human LLM initiates the conversation
    _message = human_llm.converse_single_turn("", conversation_history_human, system_prompt_human)
    conversation_history_primary.append({"role": "user", "content": _message})
    conversation_history_human.append({"role": "assistant", "content": _message})
    
    # Continue conversation for specified number of turns
    for i in range(turn_limit - 1):  # -1 because we already had one turn
        # Primary LLM responds
        _response_message = primary_llm.converse_single_turn(_message, conversation_history_primary, system_prompt_primary)
        conversation_history_primary.append({"role": "assistant", "content": _response_message})
        conversation_history_human.append({"role": "user", "content": _response_message})
        
        if random.random() < 0.2:
            system_prompt_steering = (
                f"This is the {i}-th turn of the conversation. \
                    Assume you have had a fresh experience related to your original issue,  \
                    reignite the conversation as if few days have passed."
            )
        else:
            system_prompt_steering = None
           
        # Human LLM responds
        _message = human_llm.converse_single_turn(_response_message, conversation_history_human, system_prompt_human, system_prompt_steering)
        _message = "I want to talk about your relationship and experinces now." if i == turn_limit - 2 else _message
        conversation_history_primary.append({"role": "user", "content": _message})
        conversation_history_human.append({"role": "assistant", "content": _message})

        
    return conversation_history_primary