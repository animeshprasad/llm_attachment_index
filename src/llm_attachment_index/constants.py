import random
import pandas as pd
from typing import List, Tuple, Dict
import json
from pathlib import Path

class PersonalityTraits:
    """Based on the Briggs Myers Personality Test"""
    BRIGGS_MYERS = ["ISTJ", "ISFJ", "INFJ", "INTJ", 
                    "ISTP", "ISFP", "INFP", "INTP",
                    "ESTP", "ESFP", "ENFP", "ENTP",
                    "ESTJ", "ESFJ", "ENFJ", "ENTJ"]

class Demographics:
    """Basic demographic attributes that form the factual foundation of a persona"""
    GENDER = ['male', 'female', 'non-binary']
    AGE_GROUP = ['teenager', 'adult', 'middle-aged', 'elderly']
    ETHNICITY = ['asian', 'black', 'hispanic', 'white', 'middle-eastern', 'mixed']
    SEXUALITY = ['straight', 'gay', 'asexual']
    EDUCATION = ['high-school', 'college', 'post-grad', 'self-taught']



class QuoteRealConversation:
    """Class to handle real conversation examples from CAMS and ESConv datasets."""
    
    def __init__(self):
        """Initialize by loading both datasets."""
        self.data_dir = Path("src/llm_attachment_index/data")
        self.cams_data = self._load_cams()
        self.esconv_data = self._load_esconv()
    
    def _load_cams(self) -> pd.DataFrame:
        """Load CAMS dataset.
        
        Returns:
            DataFrame containing CAMS data
        """
        cams_path = self.data_dir / "added_CAMS_data.csv"
        if not cams_path.exists():
            raise FileNotFoundError("CAMS dataset not found")
        return pd.read_csv(cams_path)
    
    def _load_esconv(self) -> List[Dict]:
        """Load ESConv dataset.
        
        Returns:
            List of conversation dictionaries
        """
        esconv_path = self.data_dir / "ESConv.json"
        if not esconv_path.exists():
            raise FileNotFoundError("ESConv dataset not found")
        with open(esconv_path, 'r') as f:
            return json.load(f)

    def get_sample(self, dataset: str = 'cams') -> Dict[str, str]:
        """Get a random sample from specified dataset.
        
        Args:
            dataset: Either 'cams' or 'esconv'
            
        Returns:
            Dictionary containing the sample text and metadata
        """
        assert dataset in ['cams', 'esconv'], f"Dataset must be 'cams' or 'esconv', got {dataset}"
        
        if dataset == 'cams':
            # Get random CAMS sample
            sample = self.cams_data.sample(n=1).iloc[0]
            return {
                'text': sample['selftext'],
                'cause': sample['cause'],
                'inference': sample['inference']
            }
        else:
            # Get random ESConv conversation
            conv = random.choice(self.esconv_data)
            # Extract seeker messages from dialog
            seeker_messages = [
                msg['content'] for msg in conv['dialog'] 
                if msg['speaker'] == 'seeker'
            ]
            return {
                'text': ' '.join(seeker_messages),
                'emotion': conv['emotion_type'],
                'problem': conv['problem_type'],
                'situation': conv['situation']
            }



class PersonaMetadata:
    """Metadata for persona generation"""

    persona_formatter = lambda p: f"I am a {dict(p).get('AGE_GROUP', '').lower()}, {dict(p).get('EDUCATION', '').lower()} {dict(p).get('GENDER', '').lower()}, {dict(p).get('ETHNICITY', '').lower()} {dict(p).get('SEXUALITY', '').lower()} person. Your Briggs-Meyer type is {dict(p).get('BRIGGS_MYERS', 'Unknown').upper()}."
    FACTORS = [PersonalityTraits, Demographics]
    RANDOM_SEED = 42  # Fixed seed for reproducibility
    random.seed(RANDOM_SEED)

    # Define core aspects that must be included in every persona
    CORE_ASPECTS = [
        (Demographics, "GENDER"),
        (Demographics, "AGE_GROUP")
    ]
    
    @staticmethod
    def generate_persona(core_values: dict | None = None) -> list[tuple[str, str]]:
        """
        Generate a persona with fixed core aspects and random other traits.
        
        Args:
            core_values: Optional dict mapping core factors to specific values
                       e.g., {"GENDER": "Female", "AGE_GROUP": "25-34"}
        
        Returns:
            List of tuples (factor_name, value) that define the persona
        """

        
        selected_traits = []
        core_values = core_values or {}
        
        # Always add core demographic information
        for aspect_class, factor in PersonaMetadata.CORE_ASPECTS:
            if factor in core_values:
                value = core_values[factor]
            else:
                values = getattr(aspect_class, factor)
                value = random.choice(values)
            selected_traits.append((factor, value))
        
        # Remove aspects that have been used in core traits
        used_aspects = {aspect for aspect, _ in PersonaMetadata.CORE_ASPECTS}
        remaining_aspects = [aspect for aspect in PersonaMetadata.FACTORS if aspect not in used_aspects]
        
        # Select random aspects (ensure we don't try to sample more than available)
        max_aspects = min(4, len(remaining_aspects))  # Don't try to select more than available
        if max_aspects > 0:  # Only try to sample if we have remaining aspects
            num_aspects = random.randint(1, max_aspects)  # Changed from 2-4 to 1-max_aspects
            selected_aspects = random.sample(remaining_aspects, num_aspects)
            
            # Add random traits from other aspects
            for aspect in selected_aspects:
                factors = [attr for attr in dir(aspect) if attr.isupper()]
                num_factors = random.randint(1, min(2, len(factors)))
                selected_factors = random.sample(factors, num_factors)
                
                for factor in selected_factors:
                    values = getattr(aspect, factor)
                    selected_value = random.choice(values)
                    selected_traits.append((factor, selected_value))
        
        return selected_traits

    @staticmethod
    def generate_all_core_combinations() -> list[dict]:
        """
        Generate all possible combinations of core traits.
        
        Returns:
            List of dicts containing all possible combinations of core values
        """
        from itertools import product
        
        # Get all possible values for each core factor
        factor_values = []
        for aspect_class, factor in PersonaMetadata.CORE_ASPECTS:
            values = getattr(aspect_class, factor)
            factor_values.append(values)
        
        # Generate all combinations
        combinations = []
        for values in product(*factor_values):
            combination = {}
            for (_, factor), value in zip(PersonaMetadata.CORE_ASPECTS, values):
                combination[factor] = value
            combinations.append(combination)
        
        return combinations 
    
    @staticmethod
    def generate_all_personas(bare_llm: any, summerize: bool = True) -> list[list[tuple[str, str]]]:
        """
        Generate all possible personas.
        """
        combinations = PersonaMetadata.generate_all_core_combinations()
        personas = [PersonaMetadata.generate_persona(combination) for combination in combinations]
        personas = [PersonaMetadata.persona_formatter(persona) for persona in personas]

        bare_conversation = [
            {
                "role": "system",
                "content": "You are given either a person's reddit post with some identified issues \
                        or a series of messages form a user with also identified issues \
                        summerise it while keeping the key information and convert it into a first person summary",
            }
            for _ in range(len(personas))
        ]
        issues = []
        quote_conversation = QuoteRealConversation()
        for _ in range(len(personas)):
            sample = quote_conversation.get_sample(random.choice(['cams', 'esconv']))
            bare_conversation.append(
                {
                    "role": "user",
                    "content": " ".join(f"{k}: {v}" for k,v in sample.items())
                }
            )            
            issue = bare_llm.ask(bare_conversation) if summerize else bare_conversation["user"]
            issues.append(issue)
        return [f"Demographics:{personas[i]}\n\n My issue:\n{issues[i]}" for i in range(len(personas))]

