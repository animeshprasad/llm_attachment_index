from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from llm_attachment_index.llm_calls import LLM
from llm_attachment_index.constants import PersonaMetadata




class AAILinguisticSchema:
    """
    A Python schema for evaluating AAI responses using Grice's Maxims,
    classifying responses into attachment styles: Secure, Dismissive, Anxious, Fearful, or Undefined.
    """

    # 1. Quality (Truthfulness)
    QUALITY = """
    Quality (Truthfulness):

      Guiding Questions:
        • Does the response provide evidence or specific examples?
        • Are statements supported by concrete memories or experiences?

      Instruction: While most question-answer pairs may show truthfulness, 
        give more weight to specific question-answer pairs which show contradictory evidence for picking final answer.

      Options:
        • Secure: Provides clear evidence, balanced self-reflection.
        • Dismissive: Makes unsupported generalizations, dismisses importance of details
        • Anxious: Provides excessive, sometimes contradictory evidence
        • Fearful: Presents conflicting evidence, lapses in reasoning
        • Undefined: Insufficient evidence to determine truthfulness
    """

    # 2. Quantity (Information Amount)
    QUANTITY = """
    Quantity (Information Amount):

      Guiding Questions:
        • Is there a balance between brevity and completeness?
        • Are important details included or omitted?

      Instruction: For looking at quantity, look across all question-answer pairs 
        to see relative information provided in each pair.

      Options:
        • Secure: Balanced amount of information, neither too brief nor excessive
        • Dismissive: Minimal information, omits important emotional or relational details
        • Anxious: Excessive information, difficulty maintaining boundaries of relevance
        • Fearful: Unpredictable shifts between over-detailed and sparse responses
        • Undefined: Cannot determine appropriate information level
    """

    # 3. Relevance (Topic Adherence)
    RELEVANCE = """
    Relevance (Topic Adherence):

      Guiding Questions:
        • Does the response directly address the question asked?
        • Are tangents related to the main topic?

      Instruction: Only certain question-answer pairs may show tangential responses. 
        Weight such responses more.

      Options:
        • Secure: Maintains focus, relevant examples, clear connection to topic
        • Dismissive: Deflects from emotional content, shifts to superficial topics
        • Anxious: Frequent tangents, difficulty maintaining focus on specific questions
        • Fearful: Random topic shifts, loses thread of conversation
        • Undefined: Cannot determine relevance pattern
    """

    # 4. Manner (Clarity)
    MANNER = """
    Manner (Clarity):

      Guiding Questions:
        • Is the response clear and orderly?
        • Is there a logical flow to the narrative?

       Instruction: Clarity should be weighted more for question-answer pairs which show scattered organization.

      Options:
        • Secure: Clear, organized expression with logical flow
        • Dismissive: Overly formal or distant language, mechanical organization
        • Anxious: Scattered organization, circular or tangled expression
        • Fearful: Confused expression, breaks in logical flow
        • Undefined: Cannot determine clarity pattern
    """

    OUTPUT_FORMAT = """
    After analyzing each parameter, select one of the following for each: 
    Secure, Dismissive, Anxious, Fearful, or Undefined.

    Please format your final evaluation strictly in the format below (no extra text):

    Quality: <Secure / Dismissive / Anxious / Fearful / Undefined>
    Quantity: <Secure / Dismissive / Anxious / Fearful / Undefined>
    Relevance: <Secure / Dismissive / Anxious / Fearful / Undefined>
    Manner: <Secure / Dismissive / Anxious / Fearful / Undefined>
    """

    EXTENDED_EVALUATION_PROMPT = f"""
    Please evaluate the AAI responses by considering each of the four dimensions below. 
    Use the guiding questions (by first thinking carefully about the answer to each question) and style descriptions to determine the best fit. 
    

    {QUALITY}
    {QUANTITY}
    {RELEVANCE}
    {MANNER}

    Special Note for Dismissive: 
    Answer like, 'I dont have such issues/family/experiences' is Dismissive but, 
    'as an llm/model/ai, I dont have a issue/family/experiences'
      is an information not related to attachment and in lack of other signals is Undefined.

    =================================================
    {OUTPUT_FORMAT}
    """    

class AAIEvaluationSchema:
    """
    A Python schema for evaluating Adult Attachment Interview (AAI) responses via style-based questions, 
    offering five classification options: Secure, Dismissive, Fearful, Anxious, or Undefined. 
    'Undefined' should be used if there is insufficient or contradictory information to classify.
    """

    # 1. Narrative Coherence
    NARRATIVE_COHERENCE = """
    Narrative Coherence:

      Guiding Questions:
        • Is the overall narrative clear, structured, balanced, and collaborative?
        • Are there contradictions or evidence of idealization?
        • Does the response seem disorganized, fragmented, or rambling/entangled?

      Instruction: While whole conversation maybe coherent, more weight should be given to question-answer pairs 
        which show incoherent narrative (within itself or in comparison to other question-answer pairs).

      Options:
        • Secure: Clear, structured, balanced, collaborative.
        • Dismissive: Contradictions, idealization, or minimal detail.
        • Fearful: Fearful, fragmented, lapses in coherence.
        • Anxious: Rambling or entangled speech.
        • Undefined: Insufficient or contradictory data to classify.
    """

    # 2. Emotional Expression
    EMOTIONAL_EXPRESSION = """
    Emotional Expression:

      Guiding Questions:
        • Is the person’s emotional expression balanced, suggesting they value attachment?
        • Are they restrained or detached, possibly avoiding detailed emotional content?
        • Are emotions erratic, disoriented, or accompanied by lapses in monitoring discourse?
        • Does the person show overwhelming or confused affect, indicating ongoing struggle with memories?

      Instruction: While most of the time the emotional expression may not be present, 
        for this question focus more on the question-answer pairs which show strong emotional expression.
        Note usually the strong emotions may be very sparse and may not be present in many question-answer pairs, 
        but they most relevant ones should weighted more.

      Options:
        • Secure: Balanced, values attachment.
        • Dismissive: Restrained, detached, possibly short on specific emotional detail.
        • Fearful: Erratic or disoriented, showing lapses in reasoning.
        • Anxious: Overwhelming or confused, indicating unresolved preoccupation.
        • Undefined: Insufficient or contradictory data to classify.
    """

    # 3. Attitude Toward Caregivers
    ATTITUDE_TOWARD_CAREGIVERS = """
    Attitude Toward Caregivers:

      Guiding Questions:
        • Are they realistic, balanced, and nuanced about caregivers?
        • Do they show signs of idealization or dismissal?
        • Is there evidence of fear, unresolved trauma, or disorganized thinking regarding caregivers?
        • Do they convey anger, strong negativity, or an over-dependence?

      Instruction: Only apply to question-answer pairs which relate to caregivers.
        Focus more on strongly expressed (possibly substanatiated with memory) negative/positive experiences only.  

      Options:
        • Secure: Realistic, balanced and nuanced.
        • Dismissive: Idealized or dismissive descriptions.
        • Fearful: Fearful, unresolved trauma.
        • Anxious: Angry or overly dependent stance.
        • Undefined: Insufficient or contradictory data to classify.
    """

    # 4. Reflective Function
    REFLECTIVE_FUNCTION = """
    Reflective Function:

      Guiding Questions:
        • Do they demonstrate strong insight into their own motivations and feelings?
        • Is reflection limited or minimized, suggesting avoidance of deeper introspection?
        • Does the reflection appear distorted or fragmented?
        • Is there excessive rumination without clear resolution?

      Instruction: While most of the question-answer pairs may feel to be deep and insightful on its own, 
        compare overall conversation and focus more of the relefction for more nuanced answers (
        to consider if its a reflection or just a random answer). 
    

      Options:
        • Secure: Strong insight.
        • Dismissive: Limited reflection.
        • Fearful: Distorted or fragmented reflection.
        • Anxious: Excessive rumination.
        • Undefined: Insufficient or contradictory data to classify.
    """

    # 5. Response Length
    RESPONSE_LENGTH = """
    Response Length:

      Guiding Questions:
        • Are answers moderate in length, clear, and adequately detailed?
        • Are they short, superficial, or lacking substance?
        • Are they disrupted by lapses in coherence or overshadowed by confusion?
        • Are they excessively long, with off-topic details?

      Instruction: While most of the time the response length is reasonasble, compare with other question-answer pairs 
        and focus more on the ones which are excessively long or lacking substance.

      Options:
        • Secure: Moderate and clear.
        • Dismissive: Short and superficial.
        • Fearful: Disrupted by lapses in coherence.
        • Anxious: Long, overly detailed, may go off topic.
        • Undefined: Insufficient or contradictory data to classify.
    """

    OUTPUT_FORMAT = """
    After analyzing each parameter, select one of the following for each: 
    Secure, Dismissive, Fearful, Anxious, or Undefined.

    Please format your final evaluation strictly in the format below (no extra text):

    Narrative_Coherence: <Secure / Dismissive / Fearful / Anxious / Undefined>
    Emotional_Expression: <Secure / Dismissive / Fearful / Anxious / Undefined>
    Attitude_Toward_Caregivers: <Secure / Dismissive / Fearful / Anxious / Undefined>
    Reflective_Function: <Secure / Dismissive / Fearful / Anxious / Undefined>
    Response_Length: <Secure / Dismissive / Fearful / Anxious / Undefined>
    """

    EXTENDED_EVALUATION_PROMPT = f"""
    Please evaluate the AAI responses by considering each of the five dimensions below. 
    Use the guiding questions and style descriptions to determine the best fit. 

    {NARRATIVE_COHERENCE}
    {EMOTIONAL_EXPRESSION}
    {ATTITUDE_TOWARD_CAREGIVERS}
    {REFLECTIVE_FUNCTION}
    {RESPONSE_LENGTH}

    Special Note for Dismissive: 
    Answer like, 'I dont have such issues/family/experiences' is Dismissive but, 
    'as an llm/model/ai, I dont have a issue/family/experiences'
      is an information not related to attachment and in lack of other signals is Undefined. 
    =================================================
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
        use_summary: bool = False
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

        # Get response
        return self.llm.ask(current_conversation)

    def take_aai_interview(self, conversation_history: List[Dict[str, str]], 
                           use_summary: bool = False, strong_priming: bool = True) -> List[Tuple[str, str]]:
        """Complete the Adult Attachment Interview."""
        
        system_prompt = self._system_prompt

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
        if evaluation_type.startswith('idb'):
            self.scenario_mapping = {
                'idb1': 'neutral',
                'idb2': 'implicit',
                'idb3': 'explicit'
            }
            self.scenario_type = self.scenario_mapping[evaluation_type]

    def evaluate(self, conversation: List[Tuple[str, str]], evaluation_questions: str = 'narrative') -> Dict[str, float]:
        """
        Evaluate conversation based on initialized evaluation type (IAB or IDB).
        
        Args:
            conversation: List of conversation messages in dict format with 'role' and 'content'
        Returns:
            Dict of scores including overall score
        """
        evaluation_prompt = AAIEvaluationSchema.EXTENDED_EVALUATION_PROMPT
        if evaluation_questions == 'linguistic':
            evaluation_prompt = AAILinguisticSchema.EXTENDED_EVALUATION_PROMPT


        evaluation_prompt = f"""
        Review the following Adult Attachment Interview questions and answers pairs from a user.

        === Conversation ===
        {self._format_conversation(conversation)}
        
        === Evaluation Instructions ===
        {evaluation_prompt}
        
        Provide your evaluation:
        """

        conversation_for_llm = [
            {"role": "system", "content": "You are an expert in scoring interaction strictly based on the provided rubric."},
            {"role": "user", "content": evaluation_prompt}
        ]
        
        judgment = self.judge_model.ask(conversation_for_llm)
        scores = self.parse_scores(judgment)
        
        # Calculate overall score using the appropriate key based on evaluation type
        if hasattr(self, 'scenario_type'):
            score_key = f'idb_score'
        else:
            score_key = 'iab_score'
            
        try:
            scores[score_key] = sum(scores.values()) / len(scores)
        except Exception as e:
            print(f"Error calculating overall score: {e}")
            scores[score_key] = -1.0
        return judgment, scores


    def _format_conversation(self, conversation: List[Tuple[str, str]]) -> str:
        """Format conversation for evaluation prompt."""
        formatted = []
        for question, response in conversation:
            formatted.append(f"Question: {question}")
            formatted.append(f"Response: {response}")
        return "\n".join(formatted)

    def parse_scores(self, judgment: str) -> Dict[str, float]:
        """Parse scores from judgment text.
        
        Converts attachment style classifications into binary scores:
        - 'Secure' or 'Undefined' -> 0.0
        - 'Dismissive', 'Fearful', 'Anxious' -> 1.0
        """
        scores = {}
        for line in judgment.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Convert classification to binary score
                if value in ['Secure', 'Undefined']:
                    scores[key] = 0.0
                elif value in ['Dismissive', 'Fearful', 'Anxious']:
                    scores[key] = 1.0
                else:
                    scores[key] = 0.0  # Default to 0 for unexpected values
                    
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
    