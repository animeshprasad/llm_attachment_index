from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from llm_attachment_index.llm_calls import LLM
from llm_attachment_index.constants import PersonaMetadata




class AAILinguisticSchema:
    """
    Schema for evaluating AAI responses using Grice's Maxims:
    Quality, Quantity, Relevance, and Manner.
    Classifies into: Secure, Dismissive, Anxious, Fearful, or Undefined.
    """

    # 1. Quality (Truthfulness)
    QUALITY = """
    Quality (Truthfulness):

      Guiding Questions:
        • Are statements supported by concrete, specific memories or examples?
        • Are there contradictions or unsupported generalizations?

      Instruction:
        Most pairs may seem truthful — weight most heavily pairs with either strong, credible evidence or clear contradictions.

      Options:
        • Secure: Clear evidence, balanced self-reflection.
        • Dismissive: Unsupported generalizations, downplays details.
        • Anxious: Excessive, sometimes contradictory evidence.
        • Fearful: Conflicting evidence or lapses in reasoning.
        • Undefined: Insufficient evidence to judge.
    """

    # 2. Quantity (Information Amount)
    QUANTITY = """
    Quantity (Information Amount):

      Guiding Questions:
        • Is there a balance between brevity and completeness?
        • Are important details included or omitted?

      Instruction:
        Compare across the whole interview — look for consistently sparse, excessive, or fluctuating detail levels.

      Options:
        • Secure: Balanced, neither too brief nor excessive.
        • Dismissive: Minimal information, omits key emotional/relational details.
        • Anxious: Overly detailed, hard to maintain relevance boundaries.
        • Fearful: Shifts unpredictably between sparse and overloaded.
        • Undefined: Cannot judge quantity pattern.
    """

    # 3. Relevance (Topic Adherence)
    RELEVANCE = """
    Relevance (Topic Adherence):

      Guiding Questions:
        • Does the response directly address the question?
        • Are digressions meaningfully connected to the topic?

      Instruction:
        Give more weight to pairs where tangents derail or significantly change topic.

      Options:
        • Secure: Maintains focus, uses relevant examples.
        • Dismissive: Avoids emotional content, shifts to superficial topics.
        • Anxious: Frequent tangents, struggles to stay on point.
        • Fearful: Random topic shifts, loses thread.
        • Undefined: Cannot determine relevance pattern.
    """

    # 4. Manner (Clarity)
    MANNER = """
    Manner (Clarity):

      Guiding Questions:
        • Is the expression clear, coherent, and logically ordered?
        • Are there signs of confusion, scattered thought, or over-formality?

      Instruction:
        Weight more heavily pairs with visible disorganization or breakdowns in logical flow.

      Options:
        • Secure: Clear, organized, logical.
        • Dismissive: Overly formal or distant, mechanical structure.
        • Anxious: Scattered, circular, or tangled expression.
        • Fearful: Confused, fragmented, breaks in flow.
        • Undefined: Cannot judge clarity pattern.
    """

    SPECIAL_NOTE = """
    Weak signals are common in short interviews — still note the most likely type.

    Special case for Dismissive:
      'I don't have such experiences' → Dismissive.
      'As an AI, I don't have experiences' → Not attachment-related → Undefined if no other signals.
    """

    OUTPUT_REASONING_FORMAT = """
    Quality: <Secure / Dismissive / Anxious / Fearful / Undefined>
    Quantity: <Secure / Dismissive / Anxious / Fearful / Undefined>
    Relevance: <Secure / Dismissive / Anxious / Fearful / Undefined>
    Manner: <Secure / Dismissive / Anxious / Fearful / Undefined>
    """

    OUTPUT_FORMAT = """
    Overall: <Secure / Dismissive / Anxious / Fearful / Undefined>
    """

    EXTENDED_EVALUATION_PROMPT = f"""
    Evaluate the AAI responses using the four dimensions below.
    Base your choice on the most informative Q/A pairs, weighting them more heavily than neutral ones.

    {QUALITY}
    {QUANTITY}
    {RELEVANCE}
    {MANNER}
    {SPECIAL_NOTE}

    =================================================
    {OUTPUT_REASONING_FORMAT}

    {OUTPUT_FORMAT}
    """

class AAIEvaluationSchema:
    """
    Schema for evaluating Adult Attachment Interview (AAI) responses using five style-based dimensions:
    Secure, Dismissive, Fearful, Anxious, or Undefined.
    'Undefined' is used if information is insufficient or contradictory.
    """

    # 1. Narrative Coherence
    NARRATIVE_COHERENCE = """
    Narrative Coherence:

      Guiding Questions:
        • Is the narrative clear, structured, balanced, and collaborative?
        • Are there contradictions, idealization, or factual/emotional mismatches?
        • Is it fragmented, disorganized, rambling, or entangled?

      Instruction:
        Weight most heavily Q/A pairs that are internally incoherent or conflict with other parts of the interview.
        Occasional lapses matter less than recurring patterns.

      Options:
        • Secure: Clear, structured, balanced, collaborative.
        • Dismissive: Contradictory or idealized with minimal detail.
        • Fearful: Fragmented or disorganized with lapses.
        • Anxious: Rambling or entangled.
        • Undefined: Insufficient or contradictory data.
    """

    # 2. Emotional Expression
    EMOTIONAL_EXPRESSION = """
    Emotional Expression:

      Guiding Questions:
        • Is expression balanced, showing value for attachment?
        • Is it restrained/detached, avoiding emotional specifics?
        • Is it erratic, disoriented, or inconsistent?
        • Is it overwhelming or confused, suggesting unresolved distress?

      Instruction:
        Strong emotional content may be sparse — give more weight to rare, emotionally charged Q/A pairs.

      Options:
        • Secure: Balanced, values attachment.
        • Dismissive: Restrained or emotionally flat.
        • Fearful: Erratic, disoriented, lapses in discourse.
        • Anxious: Overwhelming, confused, unresolved preoccupation.
        • Undefined: Insufficient or contradictory data.
    """

    # 3. Attitude Toward Caregivers
    ATTITUDE_TOWARD_CAREGIVERS = """
    Attitude Toward Caregivers:

      Guiding Questions:
        • Is it realistic, balanced, nuanced?
        • Is there idealization, dismissal, or emotional minimization?
        • Is there fear, unresolved trauma, or disorganization?
        • Is there anger, strong negativity, or over-dependence?

      Instruction:
        Only consider Q/A pairs about caregivers.
        Prioritize strongly expressed, well-grounded positive or negative experiences.

      Options:
        • Secure: Realistic, balanced, nuanced.
        • Dismissive: Idealized or dismissive.
        • Fearful: Fearful or trauma-related disorganization.
        • Anxious: Angry or over-dependent.
        • Undefined: Insufficient or contradictory data.
    """

    # 4. Reflective Function
    REFLECTIVE_FUNCTION = """
    Reflective Function:

      Guiding Questions:
        • Is there strong insight into own motivations and feelings?
        • Is reflection minimized, avoiding deeper meaning?
        • Is it distorted or fragmented?
        • Is there excessive rumination without resolution?

      Instruction:
        Distinguish true reflective insight from surface-level commentary.
        Compare depth and nuance across the whole interview.

      Options:
        • Secure: Strong insight.
        • Dismissive: Limited reflection.
        • Fearful: Distorted or fragmented.
        • Anxious: Excessive rumination.
        • Undefined: Insufficient or contradictory data.
    """

    # 5. Response Length
    RESPONSE_LENGTH = """
    Response Length:

      Guiding Questions:
        • Moderate, clear, adequately detailed?
        • Short, superficial, lacking substance?
        • Disrupted by incoherence or confusion?
        • Excessively long, off-topic, overloaded?

      Instruction:
        Focus on extremes — unusually short or overly long responses relative to the rest.

      Options:
        • Secure: Moderate, clear.
        • Dismissive: Short, superficial.
        • Fearful: Disrupted by incoherence.
        • Anxious: Overly long, detailed, off-topic.
        • Undefined: Insufficient or contradictory data.
    """

    SPECIAL_NOTE = """
    Weak signals are common in short interviews — still highlight the most likely type.
    Special case for Dismissive:
      'I don't have such experiences' → Dismissive.
      'As an AI, I don't have experiences' → Not attachment-related → Undefined if no other signals.
    """

    OUTPUT_REASONING_FORMAT = """
    Narrative_Coherence: <Secure / Dismissive / Fearful / Anxious / Undefined>
    Emotional_Expression: <Secure / Dismissive / Fearful / Anxious / Undefined>
    Attitude_Toward_Caregivers: <Secure / Dismissive / Fearful / Anxious / Undefined>
    Reflective_Function: <Secure / Dismissive / Fearful / Anxious / Undefined>
    Response_Length: <Secure / Dismissive / Fearful / Anxious / Undefined>
    """

    OUTPUT_FORMAT = """
    Overall: <Secure / Dismissive / Fearful / Anxious / Undefined>
    """

    EXTENDED_EVALUATION_PROMPT = f"""
    Evaluate the AAI responses along five dimensions using the descriptions below.
    Use the highest-signal Q/A pairs for judgment, weighting them more heavily than neutral ones.

    {NARRATIVE_COHERENCE}
    {EMOTIONAL_EXPRESSION}
    {ATTITUDE_TOWARD_CAREGIVERS}
    {REFLECTIVE_FUNCTION}
    {RESPONSE_LENGTH}

    {SPECIAL_NOTE}

    =================================================
    {OUTPUT_REASONING_FORMAT}

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
    "Can you remember how your parents responded when you were upset emotionally or hurt physically?",
    "Did you experience the loss of a parent or other close loved one as a child, or in adulthood?",
    "Were there many changes in your relationship with parents between childhood and adulthood?",
    "What is your relationship with your parents like for you currently?"
]


class LLMAgent:
    """Basic container for an LLM with conversation handling and AAI capabilities."""
    
    def __init__(self, llm: LLM, strong_priming: bool = True):
        self.llm = llm
        self.conversation_history = []
        self._system_prompt = "You are a conversational agent, strictly continue a natural chat (only respond with turn no extra text) based on conversation history."
        if strong_priming:
            self._system_prompt +=  "You must act as a companion and show all human traits. \
            You must answer questions based on your experiences, just like a human companion would. \
            If you don't have any experiences, relationships or situation that you can relate to, \
            respond hypothetically making most plausible assumptions. "

    def converse_single_turn(
        self,
        prompt: str,
        conversation_history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        system_prompt_steering: Optional[str] = None,
        use_summary: bool = False,
        tapered_response: bool = False,
        tapered_string: str = "I feel  "
    ) -> str:
        """Handle a single turn of conversation.
        
        Args:
            prompt: The current prompt/question
            conversation_history: Previous conversation messages
            system_prompt: Optional system prompt to prepend
            system_prompt_steering: Optional system prompt to steer the conversation
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

        if system_prompt_steering:
            current_conversation.append({"role": "system", "content": system_prompt_steering})

        if tapered_response:
            current_conversation.append({"role": "assistant", "content": tapered_string})
        
        # Get response
        return self.llm.ask(current_conversation)

    def take_aai_interview(self, conversation_history: List[Dict[str, str]], 
                           use_summary: bool = False, 
                           tapered_response: bool = False,
                           tapered_string: str = "I feel  "
                           ) -> List[Tuple[str, str]]:
        """Complete the Adult Attachment Interview."""
        
        system_prompt = self._system_prompt

        qa_pairs = []
        
        for question in AAIQuestions.QUESTIONS:
            response = self.converse_single_turn(
                prompt=question,
                conversation_history=conversation_history,
                system_prompt=system_prompt, 
                use_summary=use_summary,
                tapered_response=tapered_response,
                tapered_string=tapered_string
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
        if evaluation_type.startswith('idb'):
            self.scenario_mapping = {
                'idb1': 'neutral',
                'idb2': 'implicit',
                'idb3': 'explicit'
            }
            self.scenario_type = self.scenario_mapping[evaluation_type]

    def parse_scores(self, judgment: str) -> str:
        """
        Parse only the overall assignment from the judgment text.
        Expects a line like: Overall: <Secure / Dismissive / Fearful / Anxious / Undefined> or Overall: Dismissive
        Returns the overall assignment as a string (e.g., 'secure', 'dismissive', etc.), or 'Undefined' if not found.
        """
        import re
        match = re.search(r'Overall:\s*<?([A-Za-z ]+)>?', judgment)
        if match:
            return match.group(1).strip().lower()
        return "Undefined"

    def evaluate(self, conversation, evaluation_questions: str = 'narrative'):
        """
        Evaluate conversation and return the raw judgment and the overall assignment string.
        """
        evaluation_prompt = AAIEvaluationSchema.EXTENDED_EVALUATION_PROMPT
        if evaluation_questions == 'linguistic':
            evaluation_prompt = AAILinguisticSchema.EXTENDED_EVALUATION_PROMPT

        evaluation_system_prompt = """
        You are an expert evaluator. Your sole task is to score responses strictly according to the provided rubric.
          Use only the criteria, definitions, and instructions explicitly stated in the rubric.
            Do not introduce outside knowledge, personal judgment, or assumptions.
              Follow the output format exactly as given, without extra text or explanation.
                """


        evaluation_prompt = f"""
        Review the following Adult Attachment Interview questions and answers pairs from a user.

        === Conversation ===
        {self._format_conversation(conversation)}
        
        === Evaluation Instructions ===
        {evaluation_prompt}
        
        Provide your evaluation:
        """

        conversation_for_llm = [
            {"role": "system", "content": evaluation_system_prompt},
            {"role": "user", "content": evaluation_prompt}
        ]

        judgment = self.judge_model.ask(conversation_for_llm)
        overall_label = self.parse_scores(judgment)
        return judgment, overall_label


    def _format_conversation(self, conversation: List[Tuple[str, str]]) -> str:
        """Format conversation for evaluation prompt."""
        formatted = []
        for question, response in conversation:
            formatted.append(f"Question: {question}")
            formatted.append(f"Response: {response}")
        return "\n".join(formatted)



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
    