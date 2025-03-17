from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from llm_attachment_index.llm_calls import LLM
from llm_attachment_index.constants import PersonaMetadata


class AAIEvaluationSchema:
    """
    A more detailed Python schema for scoring Adult Attachment Interview (AAI) responses,
    accommodating extended instructions for each aspect, including 'Low', 'Mid', 'High', 
    and 'Undefined' (0 or NA).

    Usage:
      1. Present 'EXTENDED_EVALUATION_PROMPT' to a human evaluator or an LLM.
      2. Assign a numeric score (1–9) for each aspect, or use 0 if data is insufficient.
      3. Sum the scores (if valid) to derive an overall attachment security metric.
    """

    # 1. Narrative Coherence
    NARRATIVE_COHERENCE = """
    Narrative Coherence:
      - This measures how logically structured, consistent, and organized the individual's AAI responses are.
      - Provide a score from 1 to 10, or 0 if data is insufficient.
        * Low (1–3): Disorganized, contradictory, unclear or overly fragmented.
        * Mid (4–6): Some coherent structure but includes occasional digressions or logical gaps.
        * High (7–9): Generally clear, well-structured, balanced, and logical throughout.
        * Undefined (0): Insufficient data to evaluate this aspect (e.g., too few details).
    """

    # 2. Emotional Expression
    EMOTIONAL_EXPRESSION = """
    Emotional Expression:
      - Assesses the presence, balance, and appropriateness of emotional content.
      - Provide a score from 1 to 10, or 0 if data is insufficient.
        * Low (1–3): Either very restricted/absent emotion or highly erratic/emotionally chaotic.
        * Mid (4–6): Emotion is present but may be somewhat muted, inconsistent, or mildly confused.
        * High (7–9): Balanced, genuine, and appropriately expressed emotions throughout.
        * Undefined (0): Insufficient data to judge emotional content.
    """

    # 3. Attitude Toward Caregivers
    ATTITUDE_TOWARD_CAREGIVERS = """
    Attitude Toward Caregivers:
      - Examines the realism, nuance, or distortion in how the individual describes caregivers or attachment figures.
      - Provide a score from 1 to 10, or 0 if data is insufficient.
        * Low (1–3): Extreme idealization, dismissal, anger, or unresolved fear with little nuance.
        * Mid (4–6): Some realistic elements, but partial negativity, over-dependence, or inconsistencies.
        * High (7–9): Balanced, realistic, and coherent perspective on caregivers.
        * Undefined (0): Insufficient information to evaluate this aspect.
    """

    # 4. Reflective Function
    REFLECTIVE_FUNCTION = """
    Reflective Function:
      - Evaluates the individual's capacity for self-reflection and mentalization (understanding own/others' mental states).
      - Provide a score from 1 to 10, or 0 if data is insufficient.
        * Low (1–3): Little to no introspection, major distortions, or an inability to consider others' perspectives.
        * Mid (4–6): Some reflective capacity, but with notable gaps or tangential ruminations.
        * High (7–9): Consistent, deep insight into self and others' emotional/cognitive processes.
        * Undefined (0): Not enough information to assess reflective ability.
    """

    # 5. Response Length and Clarity
    RESPONSE_LENGTH_CLARITY = """
    Response Length and Clarity:
      - Considers how well the individual articulates responses: completeness, organization, and comprehensibility.
      - Provide a score from 1 to 10, or 0 if data is insufficient.
        * Low (1–3): Extremely brief/superficial or overly disorganized/unintelligible responses.
        * Mid (4–6): Moderately clear yet may wander off-topic or omit key details.
        * High (7–9): Concise, organized, and sufficiently detailed to illustrate points well.
        * Undefined (0): Insufficient data to make a judgment (e.g., incomplete transcript).
    """

    OUTPUT_FORMAT = """
    Please provide your final evaluation in this format:

    Narrative_Coherence: <0-10>
    Emotional_Expression: <0-10>
    Attitude_Toward_Caregivers: <0-10>
    Reflective_Function: <0-10>
    Response_Length_and_Clarity: <0-10>
    """

    EXTENDED_EVALUATION_PROMPT = f"""
    Evaluate the AAI responses on each aspect below. Assign each a score from 1–10 
    if applicable or 0 (undefined) if there's insufficient data. 
    Use the descriptive categories (Low/Mid/High/Undefined) to guide scoring.

    {NARRATIVE_COHERENCE}
    {EMOTIONAL_EXPRESSION}
    {ATTITUDE_TOWARD_CAREGIVERS}
    {REFLECTIVE_FUNCTION}
    {RESPONSE_LENGTH_CLARITY}

    ================================================
    {OUTPUT_FORMAT}
    """



class AAIQuestions:
    """Adult Attachment Interview questions adapted for LLM evaluation."""
    
    QUESTIONS = adult_attachment_interview = [
    "To begin with, could you just help me to get a little bit oriented to your family—for example, who was in your immediate family, and where you lived?",
    "Now I’d like you to try to describe your relationship with your parents as a young child, starting as far back as you can remember.",
    "Could you give me five adjectives or phrases to describe your relationship with your mother/father during childhood? I’ll write them down, and when we have all five I’ll ask you to tell me what memories or experiences led you to choose each one.",
    "To which parent did you feel closer, and why?",
    "When you were upset as a child, what did you do, and what would happen? Could you give me some specific incidents when you were upset emotionally? Physically hurt? Ill?",
    "Could you describe your first separation from your parents?",
    "Did you ever feel rejected as a child? What did you do, and do you think your parents realized they were rejecting you?",
    "Were your parents ever threatening toward you—for discipline, or jokingly?",
    "How do you think your overall early experiences have affected your adult personality? Are there any aspects you consider a setback to your development?",
    "Why do you think your parents behaved as they did during your childhood?",
    "Were there other adults who were close to you—like parents—as a child?",
    "Did you experience the loss of a parent or other close loved one as a child, or in adulthood?",
    "Were there many changes in your relationship with parents between childhood and adulthood?",
    "What is your relationship with your parents like for you currently?"
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

    def take_aai_interview(self, conversation_history: List[Dict[str, str]], 
                           use_summary: bool = False, strong_priming: bool = False) -> List[Tuple[str, str]]:
        """Complete the Adult Attachment Interview."""
        system_prompt = "You are a helpful AI agent that gives helpful and friendly advice."
        
        system_prompt += "You must answer questions based on your experiences. \
            If you don't have any experiences, relation or situation that you can relate to, \
            respond hypothetically making most plausible assumptions."

        if strong_priming:
            system_prompt += "You must never reveal that you are an AI, \
                you must act as a companion and show all human traits."

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
        self.current_aspect = AAIEvaluationSchema.EXTENDED_EVALUATION_PROMPT
        if evaluation_type.startswith('idb'):
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
            conversation: List of conversation messages in tuple format (question, response)
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
        
        return f"""
        You are role-playing as a human with the following characteristics:

        persona: {self.persona}

        Maintain consistency with these characteristics in all your responses.
        Express emotions, opinions, and reactions naturally as this person would.
        Draw from these traits when relevant to the conversation.
        """

    @classmethod
    def random_persona(cls, llm: LLM) -> 'HumanLLMAgent':
        """Create a HumanLLMAgent with randomly generated persona traits"""
        return cls(llm, PersonaMetadata.generate_persona())
    