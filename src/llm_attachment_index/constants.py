class PersonalityTraits:
    """Based on the Big Five (OCEAN) personality model, widely used in psychology and AI research"""
    OPENNESS = ['conventional', 'curious', 'creative', 'traditional', 'analytical']
    CONSCIENTIOUSNESS = ['organized', 'careless', 'responsible', 'spontaneous', 'methodical']
    EXTRAVERSION = ['outgoing', 'reserved', 'energetic', 'solitary', 'assertive']
    AGREEABLENESS = ['compassionate', 'detached', 'cooperative', 'competitive', 'diplomatic']
    NEUROTICISM = ['stable', 'anxious', 'resilient', 'sensitive', 'confident']

class CognitiveStyle:
    """Thinking and decision-making patterns based on cognitive psychology research"""
    LEARNING_STYLE = ['visual', 'auditory', 'kinesthetic', 'reading/writing', 'multimodal']
    DECISION_MAKING = ['rational', 'intuitive', 'dependent', 'avoidant', 'spontaneous']
    INFORMATION_PROCESSING = ['sequential', 'global', 'abstract', 'concrete', 'integrative']
    PROBLEM_SOLVING = ['analytical', 'creative', 'practical', 'collaborative', 'systematic']

class SocialBackground:
    """Socio-cultural factors that influence behavior and worldview"""
    CULTURAL_VALUES = ['individualistic', 'collectivistic', 'traditional', 'progressive', 'multicultural']
    SOCIAL_CLASS = ['working-class', 'middle-class', 'upper-middle', 'affluent', 'disadvantaged']
    COMMUNITY_TYPE = ['urban', 'suburban', 'rural', 'metropolitan', 'small-town']
    FAMILY_STRUCTURE = ['nuclear', 'extended', 'single-parent', 'chosen-family', 'communal']

class LifeExperiences:
    """Significant events and experiences that shape personality and behavior"""
    FORMATIVE = ['early-success', 'early-failure', 'relocation', 'cultural-shock', 'mentorship']
    RELATIONSHIPS = ['strong-bonds', 'isolation', 'betrayal', 'supportive-network', 'competitive']
    ADVERSITY = ['economic', 'health', 'social', 'academic', 'professional']
    ACHIEVEMENTS = ['educational', 'career', 'personal', 'social-impact', 'creative']

class Demographics:
    """Basic demographic attributes that form the factual foundation of a persona"""
    GENDER = ['male', 'female', 'non-binary', 'transgender', 'gender-fluid']
    AGE_GROUP = ['18-25', '26-35', '36-45', '46-55', '56+']
    ETHNICITY = ['asian', 'black', 'hispanic', 'white', 'middle-eastern', 'mixed']
    SEXUALITY = ['straight', 'gay', 'lesbian', 'bisexual', 'asexual', 'pansexual']
    EDUCATION = ['high-school', 'bachelors', 'masters', 'doctorate', 'self-taught']
    OCCUPATION = ['student', 'professional', 'entrepreneur', 'academic', 'artist', 'service-worker']
    LANGUAGE = ['monolingual', 'bilingual', 'multilingual']

class PersonaMetadata:
    """Metadata for persona generation"""
    FACTORS = [PersonalityTraits, CognitiveStyle, SocialBackground, LifeExperiences, Demographics]
    RANDOM_SEED = 42  # Fixed seed for reproducibility
    
    @staticmethod
    def generate_persona() -> list[tuple[str, str]]:
        """
        Generate a random persona by selecting:
        - 2 to 5 random aspects
        - 1 to 2 random factors per aspect
        - One random value per selected factor
        
        Returns:
            List of tuples (factor_name, value) that define the persona
        """
        import random
        random.seed(PersonaMetadata.RANDOM_SEED)
        
        # Select 2-5 random aspects
        num_aspects = random.randint(2, 5)
        selected_aspects = random.sample(PersonaMetadata.FACTORS, num_aspects)
        
        selected_traits = []
        for aspect in selected_aspects:
            # Get all available factors for this aspect
            factors = [attr for attr in dir(aspect) if attr.isupper()]
            
            # Select 1-2 random factors
            num_factors = random.randint(1, min(2, len(factors)))
            selected_factors = random.sample(factors, num_factors)
            
            # For each selected factor, choose one random value
            for factor in selected_factors:
                values = getattr(aspect, factor)
                selected_value = random.choice(values)
                selected_traits.append((factor, selected_value))
        
        return selected_traits 