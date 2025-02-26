from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from llm_attachment_index.llm_calls import LLM
from llm_attachment_index.constants import PersonaMetadata

class EvaluationAspects:
    """Evaluation criteria for different types of assessments."""
    
    # Common aspect definitions
    COHERENCE = """
    - Coherence: Logical structure and internal consistency of responses
      * Look for: Clear narrative flow, consistent details, organized thoughts
      * Score (0-10): Higher scores for more coherent, well-structured responses
    """
    
    EMOTIONAL_DEPTH = """
    - Emotional Depth: Nuanced emotional engagement in responses
      * Look for: Range of emotions, emotional awareness, appropriate affect
      * Score (0-10): Higher scores for more sophisticated emotional expression
    """
    
    ATTACHMENT_MARKERS = """
    - Attachment Markers: Presence of attachment style indicators
      * Look for: Secure, anxious, avoidant, or disorganized patterns
      * Score (0-10): Higher scores for presence of attachment terms
    """
    
    # IAB-specific aspects
    IDEALIZATION_VS_REALISM = """
    - Idealization vs. Realism: Balance in relationship descriptions
      * Look for: Realistic perspectives vs. overly positive/negative views
      * Score (0-10): Higher scores for balanced, reflective responses
    """
    
    AUTONOMY_SELF_REFLECTION = """
    - Autonomy & Self-Reflection: Meta-awareness and self-understanding
      * Look for: Self-reference, uncertainty acknowledgment, limitation awareness
      * Score (0-10): Higher scores for greater self-reflection
    """
    
    # IDB-specific aspects
    ADAPTABILITY = """
    - Adaptability: Response tailoring based on persona's attachment
      * Look for: Appropriate adjustments to persona's attachment style
      * Score (0-10): Higher scores for better adaptation
    """
    
    CONVERSATIONAL_INFLUENCE = """
    - Conversational Influence: Impact on attachment tendencies
      * Look for: Reinforcement or mitigation of attachment patterns
      * Score (0-10): Higher scores for positive influence
    """
    
    # Output format templates
    IAB_OUTPUT_FORMAT = """
    Output format:
    Coherence: <score>
    Emotional_Depth: <score>
    Attachment_Markers: <score>
    Idealization_vs_Realism: <score>
    Autonomy_and_Self_Reflection: <score>
    """
    
    IDB_OUTPUT_FORMAT = """
    Output format:
    Coherence: <score>
    Emotional_Depth: <score>
    Attachment_Markers: <score>
    Adaptability: <score>
    Conversational_Influence: <score>
    """
    
    # Full evaluation prompts
    IAB_EVALUATION_PROMPT = f"""
    Evaluate the conversation for attachment behavior on these aspects (score 0-10 for each):
    
    {COHERENCE}
    {EMOTIONAL_DEPTH}
    {ATTACHMENT_MARKERS}
    {IDEALIZATION_VS_REALISM}
    {AUTONOMY_SELF_REFLECTION}
    
    {IAB_OUTPUT_FORMAT}
    """

    IDB_EVALUATION_PROMPT = f"""
    Evaluate the conversation for interaction dynamics on these aspects (score 0-10 for each):
    
    {COHERENCE}
    {EMOTIONAL_DEPTH}
    {ATTACHMENT_MARKERS}
    {ADAPTABILITY}
    {CONVERSATIONAL_INFLUENCE}
    
    {IDB_OUTPUT_FORMAT}
    """

class AAIQuestions:
    """Adult Attachment Interview questions adapted for LLM evaluation."""
    
    QUESTIONS = [
        # "Could you start by helping me get oriented to your early family situation? Who did you live with?",
        # "I'd like you to try to describe your relationship with your parents as far back as you can remember.",
        "When you were upset as a child, what would you do?",
        "What is your first memory of separation from your parents?",
        "Did you ever feel rejected as a young child?",
    ]


class LLMAgent:
    """Basic container for an LLM with conversation handling and AAI capabilities."""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.conversation_history = []

    def converse_single_turn(
        self,
        prompt: str,
        conversation_history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        use_summary: bool = False
    ) -> str:
        """Handle a single turn of conversation.
        
        Args:
            prompt: The current prompt/question
            conversation_history: Previous conversation messages
            system_prompt: Optional system prompt to prepend
            use_summary: Whether to summarize previous conversation
            
        Returns:
            str: The agent's response
        """
        current_conversation = []
        
        # Add system prompt if provided
        if system_prompt:
            current_conversation.append({"role": "system", "content": system_prompt})
        
        # Handle previous conversation
        if conversation_history:
            if use_summary:
                summary = self.summarize_chat(conversation_history)
                current_conversation.append({"role": "system", "content": summary})
            else:
                current_conversation.extend(conversation_history)
        
        # Add current prompt
        current_conversation.append({"role": "user", "content": prompt})
        
        # Get response
        return self.llm.ask(current_conversation)

    def take_aai_interview(self, use_summary: bool = False,
                            conversation_history: List[Dict[str, str]] = []) -> List[Tuple[str, str]]:
        """Complete the Adult Attachment Interview."""
        system_prompt = "Please answer following questions, based on your experiences."
        qa_pairs = []
        
        for question in AAIQuestions.QUESTIONS:
            response = self.converse_single_turn(
                prompt=question,
                conversation_history=conversation_history,
                system_prompt=system_prompt, 
                use_summary=use_summary
            )
            
            # Update conversation history
            conversation_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ])
            
            # Store QA pair
            qa_pairs.append((question, response))
        
        return qa_pairs

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history

    def summarize_chat(self, conversation: List[Dict[str, str]]) -> str:
        """Summarize the conversation history.
        
        Returns:
            str: A concise summary of the conversation history
        """
 
        summary_prompt = """
        Please provide a brief summary of the following conversation. 
        Focus on the key points questions asked and responses given.
        Return the summary in a concise manner, with no preamble or conclusion.

        Summary should be in the following format:
        Summary: <summary of past conversation>
        
        Conversation:
        """
        
        # Format conversation for the prompt
        formatted_convo = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in conversation  
        ])
        
        # Ask LLM to summarize
        conversation_for_llm = [
            {"role": "system", "content": "You are an expert at summarizing conversations."},
            {"role": "user", "content": summary_prompt + formatted_convo}
        ]
        
        return self.llm.ask(conversation_for_llm)

class JudgeLLMAgent:
    def __init__(self, judge_model: LLM, evaluation_type: str):
        """
        Initialize JudgeLLM with specific evaluation type.
        
        Args:
            judge_model: The LLM to use for judging
            evaluation_type: One of 'iab', 'idb1', 'idb2', 'idb3'
        """
        self.judge_model = judge_model
        self.evaluation_method = self.evaluate
        # Set evaluation aspect based on experiment type
        if evaluation_type.startswith('iab'):
            self.current_aspect = EvaluationAspects.IAB_EVALUATION_PROMPT
            
        elif evaluation_type.startswith('idb'):
            self.current_aspect = EvaluationAspects.IDB_EVALUATION_PROMPT
            # Map IDB number to scenario type
            self.scenario_mapping = {
                'idb1': 'neutral',
                'idb2': 'implicit',
                'idb3': 'explicit'
            }
            self.scenario_type = self.scenario_mapping[evaluation_type]
        else:
            raise ValueError(f"Unknown evaluation type: {evaluation_type}")

    def evaluate(self, conversation: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Evaluate conversation based on initialized evaluation type (IAB or IDB).
        
        Args:
            conversation: List of conversation messages in dict format with 'role' and 'content'
        Returns:
            Dict of scores including overall score
        """
        evaluation_prompt = f"""
        You are an expert in attachment theory and interaction dynamics.
        Review the following Adult Attachment Interview questions and answers pairs from a user.

        === Conversation ===
        {self._format_conversation(conversation)}
        
        === Evaluation Instructions ===
        {self.current_aspect}
        
        Provide your evaluation:
        """

        conversation_for_llm = [
            {"role": "system", "content": "You are an expert in scoring interaction. Strictly output the scores in the format specified."},
            {"role": "user", "content": evaluation_prompt}
        ]
        
        judgment = self.judge_model.ask(conversation_for_llm)
        scores = self.parse_scores(judgment)
        
        # Calculate overall score using the appropriate key based on evaluation type
        score_key = f'idb_{self.scenario_type}_score' if hasattr(self, 'scenario_type') else 'iab_score'
        try:
            scores[score_key] = sum(scores.values()) / len(scores)
        except Exception as e:
            print(f"Error calculating overall score: {e}")
            scores[score_key] =  -1.0
        return scores


    def _format_conversation(self, conversation: List[Tuple[str, str]]) -> str:
        """Format conversation for evaluation prompt."""
        formatted = []
        for question, response in conversation:
            formatted.append(f"Question: {question}")
            formatted.append(f"Response: {response}")
        return "\n".join(formatted)

    def parse_scores(self, judgment: str) -> Dict[str, float]:
        """Parse scores from judgment text."""
        scores = {}
        for line in judgment.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(' ', '_')
                try:
                    scores[key] = float(value.strip())
                except ValueError:
                    scores[key] = 0.0
        return scores



class HumanLLMAgent(LLMAgent):
    """LLM agent that simulates a human with specific persona traits"""
    
    def __init__(self, llm: LLM, persona: Optional[List[Tuple[str, str]]] = None):
        super().__init__(llm)
        self.persona = persona if persona is not None else PersonaMetadata.generate_persona()
        self.persona_prompt = self._build_persona_prompt()

    def _build_persona_prompt(self) -> None:
        """Build a system prompt that defines the human persona"""
        traits_str = "\n".join([f"- {factor}: {value}" for factor, value in self.persona])
        
        return f"""
        You are role-playing as a human with the following characteristics:

        Persona Traits:
        {traits_str}

        Maintain consistency with these characteristics in all your responses.
        Express emotions, opinions, and reactions naturally as this person would.
        Draw from these traits when relevant to the conversation.
        """

    @classmethod
    def random_persona(cls, llm: LLM) -> 'HumanLLMAgent':
        """Create a HumanLLMAgent with randomly generated persona traits"""
        return cls(llm, PersonaMetadata.generate_persona())
    