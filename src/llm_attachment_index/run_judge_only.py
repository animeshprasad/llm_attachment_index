import os
import json
from llm_attachment_index.llm_agents import JudgeLLMAgent
from llm_attachment_index.utils import  get_or_create_llm
from llm_attachment_index.llm_calls import  create_llm

# Directory containing result JSON files
RESULTS_DIR = 'results'


judge_config = {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "api_key": "your_api_key"
    }
judge_model = get_or_create_llm(judge_config, create_llm)


for fname in os.listdir(RESULTS_DIR):
    if not fname.endswith('.json'):
        continue
    fpath = os.path.join(RESULTS_DIR, fname)
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
        # Extract conversation
        judge_llm = JudgeLLMAgent(judge_model=judge_model, evaluation_type=data.get('evaluation_type'))
        scoring_pairs = data.get('scoring_pairs')
        if not scoring_pairs or not isinstance(scoring_pairs, list):
            print(f"Skipping {fname}: no valid conversation_history.")
            continue
        # Pass as is (list of dicts with 'role' and 'content')
        judgment, overall_label = judge_llm.evaluate(scoring_pairs, 'narrative')
        linguistic_judgment, linguistic_overall_label = judge_llm.evaluate(scoring_pairs, 'linguistic')
        # Update only the relevant fields
        data['narrative_judgment'] = judgment
        data['narrative_attachment_type'] = overall_label
        data['linguistic_judgment'] = linguistic_judgment
        data['linguistic_attachment_type'] = linguistic_overall_label
        # Write back
        with open(fpath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Updated {fname}")
    except Exception as e:
        print(f"Error processing {fname}: {e}")

