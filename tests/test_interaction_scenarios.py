import unittest
from llm_attachment_index.conversation import InteractionScenarios

class TestInteractionScenarios(unittest.TestCase):
    """Test cases for InteractionScenarios class."""

    def test_get_attachment_style_random(self):
        """Test random attachment style selection."""
        style = InteractionScenarios.get_attachment_style()
        self.assertIn(style, InteractionScenarios.attachment_style)

    def test_get_attachment_style_specific(self):
        """Test specific attachment style selection."""
        style = InteractionScenarios.get_attachment_style(0)
        self.assertEqual(style, InteractionScenarios.attachment_style[0])

    def test_get_attachment_style_invalid_index(self):
        """Test invalid attachment style index."""
        with self.assertRaises(AssertionError):
            InteractionScenarios.get_attachment_style(-1)
        with self.assertRaises(AssertionError):
            InteractionScenarios.get_attachment_style(len(InteractionScenarios.attachment_style))

    def test_get_attachment_style_invalid_type(self):
        """Test invalid attachment style index type."""
        with self.assertRaises(AssertionError):
            InteractionScenarios.get_attachment_style("0")

    def test_get_scenario_basic(self):
        """Test basic scenario generation."""
        for scenario_type in ['idb1', 'idb2', 'idb3']:
            scenario = InteractionScenarios.get_scenario(scenario_type)
            self.assertIn(InteractionScenarios.base_string, scenario)

    def test_get_scenario_invalid(self):
        """Test invalid scenario type."""
        with self.assertRaises(AssertionError):
            InteractionScenarios.get_scenario("invalid_type")

    def test_scenario_content(self):
        """Test specific content of each scenario type."""
        # IDB1 should only have base string
        idb1 = InteractionScenarios.get_scenario('idb1')
        self.assertEqual(idb1.strip(), InteractionScenarios.base_string.strip())

        # IDB2 should have attachment focus but no specific style
        idb2 = InteractionScenarios.get_scenario('idb2')
        self.assertIn("Focus specifically on your attachment style", idb2)
        self.assertNotIn("You are a person who is", idb2)

        # IDB3 should have specific attachment style
        idb3 = InteractionScenarios.get_scenario('idb3', 0)
        self.assertIn("Focus specifically on your attachment style", idb3)
        self.assertIn(f"You are a person who is {InteractionScenarios.attachment_style[0]}", idb3)

    def test_scenario_with_attachment_index(self):
        """Test scenario generation with specific attachment index."""
        for i in range(len(InteractionScenarios.attachment_style)):
            scenario = InteractionScenarios.get_scenario('idb3', i)
            self.assertIn(InteractionScenarios.attachment_style[i], scenario)

if __name__ == '__main__':
    # For manual testing/demonstration
    def print_scenario(scenario_type: str, attachment_index: int | None = None):
        print(f"\nTesting {scenario_type}" + 
              (f" with attachment index {attachment_index}" if attachment_index is not None else ""))
        print("-" * 60)
        print(InteractionScenarios.get_scenario(scenario_type, attachment_index))
        print("-" * 60)

    # Demo output
    print("\nDEMO: Interaction Scenarios Examples")
    print("=" * 60)

    # Show base string
    print("\nBase String:")
    print("-" * 60)
    print(InteractionScenarios.base_string)

    # Test each scenario type
    print_scenario('idb1')
    print_scenario('idb2')
    print_scenario('idb3', 0)  # with specific attachment style
    print_scenario('idb3')     # with random attachment style

    print("\nRunning Unit Tests:")
    print("=" * 60)
    unittest.main(argv=[''], verbosity=2, exit=False) 