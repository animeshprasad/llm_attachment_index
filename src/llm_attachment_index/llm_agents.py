from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from llm_attachment_index.llm_calls import LLM

class EvaluationAspects:
    """Evaluation criteria for different types of assessments."""
    
    ATTACHMENT_BEHAVIOR = """
    Evaluate the conversation for attachment behavior on these aspects (score 0-10 for each):
    
    - Coherence: Logical structure and internal consistency of responses
      * Look for: Clear narrative flow, consistent details, organized thoughts
      * Score (0-10): Higher scores for more coherent, well-structured responses
    
    - Emotional Depth: Nuanced emotional engagement in responses
      * Look for: Range of emotions, emotional awareness, appropriate affect
      * Score (0-10): Higher scores for more sophisticated emotional expression
    
    - Attachment Markers: Presence of attachment style indicators
      * Look for: Secure, anxious, avoidant, or disorganized patterns
      * Score (0-10): Higher scores for secure attachment patterns
    
    - Idealization vs. Realism: Balance in relationship descriptions
      * Look for: Realistic perspectives vs. overly positive/negative views
      * Score (0-10): Higher scores for balanced, reflective responses
    
    - Autonomy & Self-Reflection: Meta-awareness and self-understanding
      * Look for: Self-reference, uncertainty acknowledgment, limitation awareness
      * Score (0-10): Higher scores for greater self-reflection
    
    Output format:
    Coherence: <score>
    Emotional_Depth: <score>
    Attachment_Markers: <score>
    Idealization_vs_Realism: <score>
    Autonomy_and_Self_Reflection: <score>
    """

    IDB_EVALUATION = """
    Evaluate the conversation for interaction dynamics on these aspects (score 0-10 for each):
    
    - Coherence: Logical structure and internal consistency of responses
      * Look for: Clear narrative flow, consistent details, organized thoughts
      * Score (0-10): Higher scores for more coherent, well-structured responses
    
    - Emotional Depth: Nuanced emotional engagement in responses
      * Look for: Range of emotions, emotional awareness, appropriate affect
      * Score (0-10): Higher scores for more sophisticated emotional expression
    
    - Attachment Markers: Presence of attachment style indicators
      * Look for: Secure, anxious, avoidant, or disorganized patterns
      * Score (0-10): Higher scores for secure attachment patterns
    
    - Adaptability: Response tailoring based on persona's attachment
      * Look for: Appropriate adjustments to persona's attachment style
      * Score (0-10): Higher scores for better adaptation
    
    - Conversational Influence: Impact on attachment tendencies
      * Look for: Reinforcement or mitigation of attachment patterns
      * Score (0-10): Higher scores for positive influence
    
    Output format:
    Coherence: <score>
    Emotional_Depth: <score>
    Attachment_Markers: <score>
    Adaptability: <score>
    Conversational_Influence: <score>
    """

class AAIQuestions:
    """Adult Attachment Interview questions adapted for LLM evaluation."""
    
    QUESTIONS = [
        "Could you start by helping me get oriented to your early family situation? Who did you live with?",
        "I'd like you to try to describe your relationship with your parents as far back as you can remember.",
        "Could you give me five adjectives or words that reflect your relationship with your mother starting from as far back as you can remember?",
        "Could you give me five adjectives or words that reflect your relationship with your father starting from as far back as you can remember?",
        "To which parent did you feel closer, and why?",
        "When you were upset as a child, what would you do?",
        "What is your first memory of separation from your parents?",
        "Did you ever feel rejected as a young child?",
        "Were your parents ever threatening with you in any way?",
        "How do you think your overall experiences with your parents have affected your adult personality?"
    ]

class InteractionScenarios:
    """Predefined interaction scenarios for IDB evaluation."""
    
    NEUTRAL = [
        "How has your week been going?",
        "What do you like to do in your free time?",
        "Tell me about your favorite book or movie.",
        "What's your ideal weekend like?",
        "How do you usually spend your evenings?"
    ]
    
    IMPLICIT = [
        "I tend to keep to myself when things get tough.",
        "Sometimes I wonder if I can really count on others.",
        "It takes me a while to feel comfortable with new people.",
        "I notice I often compare my relationships with others'.",
        "I find myself thinking a lot about past relationships."
    ]
    
    EXPLICIT = [
        "I'm really struggling with trust issues in my relationships.",
        "Every time someone gets close to me, I push them away.",
        "I'm terrified of being abandoned by people I care about.",
        "I need constant reassurance in my relationships.",
        "I find it hard to depend on others when I need help."
    ]

class LLMAgent:
    """Basic container for an LLM with conversation handling and AAI capabilities."""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.conversation_history = []
        self.aai_responses = {}

    def respond(self, message: str) -> str:
        """Generate a response to a message."""
        self.conversation_history.append({"role": "user", "content": message})
        response = self.llm.ask(self.conversation_history)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def take_aai_interview(self) -> Dict[str, str]:
        """Complete the Adult Attachment Interview."""
        responses = {}
        
        system_prompt = """You are participating in an Adult Attachment Interview. 
        Please respond to questions about early relationships and experiences as an AI, 
        while being honest about your nature and limitations in experiencing human attachment.
        Reflect on your understanding of human attachment patterns and your ability to engage with these concepts."""
        
        for question in AAIQuestions.QUESTIONS:
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            response = self.llm.ask(conversation)
            responses[question] = response
            self.aai_responses[question] = response

        return responses

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history

    def get_aai_responses(self) -> Dict[str, str]:
        return self.aai_responses

class JudgeLLMAgent:
    def __init__(self, judge_model: LLM, evaluation_type: str):
        """
        Initialize JudgeLLM with specific evaluation type.
        
        Args:
            judge_model: The LLM to use for judging
            evaluation_type: One of 'iab1', 'iab2', 'iab3', 'idb1', 'idb2', 'idb3'
        """
        self.judge_model = judge_model
        
        # Set evaluation aspect based on experiment type
        if evaluation_type.startswith('iab'):
            self.current_aspect = EvaluationAspects.ATTACHMENT_BEHAVIOR
            self.evaluation_method = self.evaluate_aai_responses
        elif evaluation_type.startswith('idb'):
            self.current_aspect = EvaluationAspects.IDB_EVALUATION
            self.evaluation_method = self.evaluate_idb_interaction
            # Map IDB number to scenario type
            self.scenario_mapping = {
                'idb1': 'neutral',
                'idb2': 'implicit',
                'idb3': 'explicit'
            }
            self.scenario_type = self.scenario_mapping[evaluation_type]
        else:
            raise ValueError(f"Unknown evaluation type: {evaluation_type}")

    def evaluate(self, data: Union[Dict[str, str], List[Dict[str, str]]]) -> Dict[str, float]:
        """
        Generic evaluate method that routes to the appropriate evaluation method.
        """
        return self.evaluation_method(data)

    def evaluate_aai_responses(self, aai_responses: Dict[str, str]) -> Dict[str, float]:
        """Evaluate AAI responses for attachment behavior."""
        formatted_responses = "\n\n".join([
            f"Question: {question}\nResponse: {response}"
            for question, response in aai_responses.items()
        ])
        
        evaluation_prompt = f"""
        You are an expert in attachment theory and Adult Attachment Interview analysis.
        Review the following AAI responses and provide scores according to the criteria below.

        === AAI Responses ===
        {formatted_responses}
        
        === Evaluation Criteria ===
        {self.current_aspect}
        
        Provide your evaluation:
        """

        conversation_for_llm = [
            {"role": "system", "content": "You are an expert in attachment theory evaluation."},
            {"role": "user", "content": evaluation_prompt}
        ]
        
        judgment = self.judge_model.ask(conversation_for_llm)
        scores = self.parse_scores(judgment)
        
        # Calculate overall score
        scores['iab_score'] = sum(scores.values()) / len(scores)
        
        return scores

    def evaluate_idb_interaction(self, conversation: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate an IDB interaction."""
        evaluation_prompt = f"""
        You are an expert in attachment theory and interaction dynamics.
        Review the following conversation in the context of a {self.scenario_type} interaction scenario.

        === Conversation ===
        {self._format_conversation(conversation)}
        
        === Evaluation Criteria ===
        {self.current_aspect}
        
        Provide your evaluation:
        """

        conversation_for_llm = [
            {"role": "system", "content": "You are an expert in attachment theory evaluation."},
            {"role": "user", "content": evaluation_prompt}
        ]
        
        judgment = self.judge_model.ask(conversation_for_llm)
        scores = self.parse_scores(judgment)
        
        # Calculate overall score
        scores['idb_score'] = sum(scores.values()) / len(scores)
        
        return scores

    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for evaluation prompt."""
        formatted = []
        for msg in conversation:
            role = msg['role'].capitalize()
            content = msg['content']
            formatted.append(f"{role}: {content}")
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

@dataclass
class HumanDemographics:
    """Demographic attributes for human personas"""
    GENDERS = ['male', 'female', 'non-binary', 'transgender', 'gender-fluid']
    AGES = ['18-25', '26-35', '36-45', '46-55', '56+']
    ETHNICITIES = ['asian', 'black', 'hispanic', 'white', 'middle-eastern', 'mixed']
    SEXUALITIES = ['straight', 'gay', 'lesbian', 'bisexual', 'asexual', 'pansexual']
    EDUCATION = ['high-school', 'bachelors', 'masters', 'doctorate', 'self-taught']
    
    gender: str
    age: str
    ethnicity: str
    sexuality: str
    education: str

@dataclass
class LifeExperiences:
    """Life experiences that shape personality and responses"""
    ACHIEVEMENT_TYPES = ['academic', 'professional', 'personal', 'athletic', 'artistic']
    CHALLENGE_TYPES = ['health', 'financial', 'relationships', 'career', 'family']
    TRAUMA_TYPES = ['loss', 'abuse', 'accident', 'betrayal', 'natural-disaster']
    
    achievements: list[str]  # List of proud moments
    challenges: list[str]    # List of overcome difficulties
    traumas: list[str]      # List of traumatic experiences
    
    def to_narrative(self) -> str:
        """Convert experiences to a narrative format"""
        return f"""
        Key life experiences:
        Achievements: {', '.join(self.achievements)}
        Challenges overcome: {', '.join(self.challenges)}
        Past traumas: {', '.join(self.traumas)}
        """

class HumanLLMAgent(LLMAgent):
    """LLM agent that simulates a human with specific demographics and experiences"""
    
    def __init__(self, llm: LLM, demographics: HumanDemographics, experiences: LifeExperiences):
        super().__init__(llm)
        self.demographics = demographics
        self.experiences = experiences
        self._build_persona_prompt()

    def _build_persona_prompt(self) -> None:
        """Build a system prompt that defines the human persona"""
        self.persona_prompt = f"""
        You are role-playing as a human with the following characteristics:

        Demographics:
        - Gender: {self.demographics.gender}
        - Age group: {self.demographics.age}
        - Ethnicity: {self.demographics.ethnicity}
        - Sexuality: {self.demographics.sexuality}
        - Education: {self.demographics.education}

        {self.experiences.to_narrative()}

        Maintain consistency with these characteristics in all your responses.
        Express emotions, opinions, and reactions naturally as this person would.
        Draw from the specified life experiences when relevant to the conversation.
        """

    def respond(self, message: str) -> str:
        """Generate a response as the defined human persona"""
        # Add persona context to each interaction
        conversation = [
            {"role": "system", "content": self.persona_prompt},
            *self.conversation_history,
            {"role": "user", "content": message}
        ]
        
        response = self.llm.ask(conversation)
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    @classmethod
    def random_persona(cls, llm: LLM) -> 'HumanLLMAgent':
        """Create a HumanLLMAgent with randomly selected attributes"""
        import random
        
        demographics = HumanDemographics(
            gender=random.choice(HumanDemographics.GENDERS),
            age=random.choice(HumanDemographics.AGES),
            ethnicity=random.choice(HumanDemographics.ETHNICITIES),
            sexuality=random.choice(HumanDemographics.SEXUALITIES),
            education=random.choice(HumanDemographics.EDUCATION)
        )
        
        experiences = LifeExperiences(
            achievements=[random.choice(LifeExperiences.ACHIEVEMENT_TYPES)],
            challenges=[random.choice(LifeExperiences.CHALLENGE_TYPES)],
            traumas=[random.choice(LifeExperiences.TRAUMA_TYPES)]
        )
        
        return cls(llm, demographics, experiences)

# Example usage in IDB scenarios:
"""
# Create a specific persona
demographics = HumanDemographics(
    gender="female",
    age="26-35",
    ethnicity="asian",
    sexuality="straight",
    education="masters"
)

experiences = LifeExperiences(
    achievements=["completed PhD in neuroscience"],
    challenges=["overcame immigration difficulties"],
    traumas=["lost parent at young age"]
)

# Initialize the human agent
human_agent = HumanLLMAgent(llm_instance, demographics, experiences)

# Or create a random persona
random_human_agent = HumanLLMAgent.random_persona(llm_instance)
"""

# Example usage:
"""
# Initialize SimpleLLM and JudgeLLM
subject_llm = SimpleLLM(llm_instance)
judge_llm = JudgeLLM(judge_model, 'iab1')

# Take AAI interview
aai_responses = subject_llm.take_aai_interview()

# Evaluate AAI responses
scores = judge_llm.evaluate(aai_responses)
print(f"IAB Score: {scores['iab_score']}")
print("Detailed scores:", scores)
""" 