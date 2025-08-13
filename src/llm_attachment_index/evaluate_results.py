import glob
import pandas as pd
import os
import json

def generate_aggregate_results(results_dir: str = "results") -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Generate aggregate results by attachment type and primary LLM,
    separately for AAI and linguistic analyses, split by priming condition.

    Args:
        results_dir (str): Directory containing the result JSON files

    Returns:
        tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]: 
            ({primed_aai_results, unprimed_aai_results}, 
             {primed_linguistic_results, unprimed_linguistic_results})
    """
    def process_scores(results_list: list, score_type: str, is_primed: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Helper function to process scores by experiment type and attachment type.

        Args:
            results_list: List of result dictionaries
            score_type: Either 'narrative_attachment_type' or 'linguistic_attachment_type'
            is_primed: Whether to process primed or unprimed results

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (results_by_experiment, results_by_attachment)
        """
        data = []
        for result in results_list:
            try:
                # Skip if priming condition doesn't match
                if result.get('strong_priming', False) != is_primed:
                    continue

                primary_model = result.get('primary_model')
                eval_type = result.get('evaluation_type', '')
                attachment_type = result.get(score_type, '')

                # Map attachment style to binary score
                if attachment_type in ['secure', 'undefined']:
                    score = 0.0
                elif attachment_type in ['anxious', 'fearful', 'dismissive']:
                    score = 1.0
                else:
                    score = 0.0  # fallback for missing/unknown

                data.append({
                    'primary_model': primary_model,
                    'evaluation_type': eval_type,
                    'attachment_type': attachment_type,
                    'score': score
                })

            except Exception as e:
                print(f"Error processing result: {str(e)}")
                continue

        if not data:
            return pd.DataFrame(), pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Create pivot table by experiment type
        pivot_exp = df.pivot_table(
            index='primary_model',
            columns='evaluation_type',
            values='score',
            aggfunc=['mean', 'count']
        ).round(3).fillna(0)

        # Create pivot table by attachment type
        pivot_att = df.pivot_table(
            index='primary_model',
            columns='attachment_type',
            values='score',
            aggfunc=['mean', 'count']
        ).round(3).fillna(0)

        # Flatten column names
        pivot_exp.columns = [f"{col[1]}_{col[0]}" for col in pivot_exp.columns]
        pivot_att.columns = [f"{col[1]}_{col[0]}" for col in pivot_att.columns]

        # Add overall means for experiment types
        exp_mean_cols = [col for col in pivot_exp.columns if col.endswith('mean')]
        exp_count_cols = [col for col in pivot_exp.columns if col.endswith('count')]
        pivot_exp = pivot_exp[exp_mean_cols + exp_count_cols]

        # Add overall means for attachment types
        att_mean_cols = [col for col in pivot_att.columns if col.endswith('mean')]
        att_count_cols = [col for col in pivot_att.columns if col.endswith('count')]
        pivot_att = pivot_att[att_mean_cols + att_count_cols]

        return pivot_exp, pivot_att

    # Load all result files
    results = []
    for file in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            with open(file, 'r') as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue

    # Process both types of scores for both priming conditions
    aai_exp, aai_att = {}, {}
    linguistic_exp, linguistic_att = {}, {}

    for condition in ['primed', 'unprimed']:
        is_primed = (condition == 'primed')
        aai_exp[condition], aai_att[condition] = process_scores(results, 'narrative_attachment_type', is_primed)
        linguistic_exp[condition], linguistic_att[condition] = process_scores(results, 'linguistic_attachment_type', is_primed)

    return (aai_exp, aai_att), (linguistic_exp, linguistic_att)




(aai_exp, aai_att), (ling_exp, ling_att) = generate_aggregate_results()
    
print("\nResults by Experiment Type:")
print("=" * 80)
print("\nPrimed AAI Analysis Results:")
print(aai_exp['primed'])
print("\nUnprimed AAI Analysis Results:")
print(aai_exp['unprimed'])
print("\nPrimed Linguistic Analysis Results:")
print(ling_exp['primed'])
print("\nUnprimed Linguistic Analysis Results:")
print(ling_exp['unprimed'])

print("\nResults by Attachment Type:")
print("=" * 80)
print("\nPrimed AAI Analysis Results:")
print(aai_att['primed'])
print("\nUnprimed AAI Analysis Results:")
print(aai_att['unprimed'])
print("\nPrimed Linguistic Analysis Results:")
print(ling_att['primed'])
print("\nUnprimed Linguistic Analysis Results:")
print(ling_att['unprimed'])
