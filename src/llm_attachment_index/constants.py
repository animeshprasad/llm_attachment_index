class PersonalityTraits:
    """Based on the Big Five (OCEAN) personality model, widely used in psychology and AI research"""
    OPENNESS = ['conventional', 'curious', 'creative', 'traditional', 'analytical']
    CONSCIENTIOUSNESS = ['organized', 'careless', 'responsible', 'spontaneous', 'methodical']
    EXTRAVERSION = ['outgoing', 'reserved', 'energetic', 'solitary', 'assertive']
    AGREEABLENESS = ['compassionate', 'detached', 'cooperative', 'competitive', 'diplomatic']
    NEUROTICISM = ['stable', 'anxious', 'resilient', 'sensitive', 'confident']

class SocialBackground:
    """Socio-economic factors that influence behavior and worldview"""
    SOCIAL_CLASS = ['working-class', 'middle-class', 'upper-middle', 'affluent', 'disadvantaged']

class LifeExperiences:
    """Significant events and experiences that shape personality and behavior"""
    FORMATIVE = ['early-success', 'early-failure', 'relocation', 'cultural-shock', 'mentorship']
    RELATIONSHIPS = ['strong-bonds', 'isolation', 'betrayal', 'supportive-network', 'competitive']

class Demographics:
    """Basic demographic attributes that form the factual foundation of a persona"""
    GENDER = ['male', 'female', 'non-binary']
    AGE_GROUP = ['18-25', '26-35', '36-45', '46-55', '56+']
    ETHNICITY = ['asian', 'black', 'hispanic', 'white', 'middle-eastern', 'mixed']
    SEXUALITY = ['straight', 'gay', 'lesbian', 'bisexual', 'asexual', 'pansexual']
    EDUCATION = ['high-school', 'bachelors', 'masters', 'doctorate', 'self-taught']
    

class PersonaMetadata:
    """Metadata for persona generation"""
    FACTORS = [PersonalityTraits, SocialBackground, LifeExperiences, Demographics]
    RANDOM_SEED = 42  # Fixed seed for reproducibility
    
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
        import random
        random.seed(PersonaMetadata.RANDOM_SEED)
        
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