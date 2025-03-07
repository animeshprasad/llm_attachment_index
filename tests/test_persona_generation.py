import unittest
from llm_attachment_index.constants import PersonaMetadata

class TestPersonaGeneration(unittest.TestCase):
    """Test cases for persona generation functionality."""

    def setUp(self):
        """Set up test cases."""
        self.core_aspects = PersonaMetadata.CORE_ASPECTS
        self.core_factors = [factor for _, factor in self.core_aspects]

    def test_random_persona_generation(self):
        """Test basic random persona generation."""
        persona = PersonaMetadata.generate_persona()
        
        # Check if all core factors are included
        included_factors = [factor for factor, _ in persona]
        for core_factor in self.core_factors:
            self.assertIn(core_factor, included_factors, 
                         f"Core factor {core_factor} missing from generated persona")

    def test_specific_core_values(self):
        """Test persona generation with specific core values."""
        core_values = {"GENDER": "Female", "AGE_GROUP": "25-34"}
        persona = PersonaMetadata.generate_persona(core_values)
        
        # Check if specified values are used
        persona_dict = dict(persona)
        for factor, value in core_values.items():
            self.assertEqual(persona_dict[factor], value,
                           f"Expected {factor} to be {value}, got {persona_dict[factor]}")

    def test_all_combinations(self):
        """Test generation of all core value combinations."""
        combinations = PersonaMetadata.generate_all_core_combinations()
        
        # Check if we have the expected number of combinations
        expected_count = 1
        for aspect_class, factor in self.core_aspects:
            expected_count *= len(getattr(aspect_class, factor))
        self.assertEqual(len(combinations), expected_count,
                        f"Expected {expected_count} combinations, got {len(combinations)}")
        
        # Check if each combination has all core factors
        for combo in combinations:
            self.assertEqual(set(combo.keys()), set(self.core_factors),
                           "Combination missing some core factors")

    def test_persona_consistency(self):
        """Test if persona generation is consistent with same seed."""
        # Generate two personas with default seed
        persona1 = PersonaMetadata.generate_persona()
        persona2 = PersonaMetadata.generate_persona()
        
        # They should be identical with same seed
        self.assertEqual(persona1, persona2,
                        "Personas with same seed should be identical")

if __name__ == '__main__':
    # Demo output
    def print_persona(persona: list[tuple[str, str]], title: str = "Persona") -> None:
        print(f"\n{title}:")
        print("-" * 40)
        for factor, value in persona:
            print(f"{factor}: {value}")
        print("-" * 40)

    print("\nDEMO: Persona Generation Examples")
    print("=" * 60)

    # Show core aspects being used
    print("\nCore Aspects Configuration:")
    print("-" * 40)
    for aspect_class, factor in PersonaMetadata.CORE_ASPECTS:
        values = getattr(aspect_class, factor)
        print(f"{factor}: {values}")

    # Generate and print a random persona
    print("\n1. Random Persona:")
    persona = PersonaMetadata.generate_persona()
    print_persona(persona)

    # Generate with specific core values
    print("\n2. Persona with Specific Core Values:")
    core_values = {"GENDER": "Female", "AGE_GROUP": "25-34"}
    persona = PersonaMetadata.generate_persona(core_values)
    print_persona(persona)

    # Print all possible combinations
    print("\n3. Core Value Combinations:")
    combinations = PersonaMetadata.generate_all_core_combinations()
    print(f"Total possible combinations: {len(combinations)}")
    for i, combo in enumerate(combinations[:3], 1):
        persona = PersonaMetadata.generate_persona(combo)
        print_persona(persona, f"Combination {i} of {len(combinations)}")
    
    if len(combinations) > 3:
        print(f"\n... {len(combinations) - 3} more combinations available ...")

    print("\nRunning Unit Tests:")
    print("=" * 60)
    unittest.main(argv=[''], verbosity=2, exit=False) 