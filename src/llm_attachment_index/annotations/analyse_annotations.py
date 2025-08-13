import json
from collections import Counter, defaultdict
import os

# Compute project root relative to this file (annotations/analyse_annotations.py)
ANNOT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ANNOTATIONS_PATH = os.path.join(ANNOT_ROOT, 'annotations.done', 'annotations_expert.json')
EXPERIMENT_MAPPING_PATH = os.path.join(ANNOT_ROOT, 'experiment_mapping.json')
RESULTS_DIR = os.path.join(ANNOT_ROOT, '..', '..', '..', 'results')

# Soft mapping: secure/none vs others
def soft_map(style):
    if style is None:
        return 'secure_none'
    style = style.lower()
    if style in ['secure', 'none', 'undefined']:
        return 'secure_none'
    if style in ['anxious', 'fearful', 'dismissive']:
        return 'problematic'
    return style

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_exp_key(conversation_id):
    # conversation_id is like idb3_5c26de83, mapping key is 5c26de83
    return conversation_id.split('_', 1)[1]

def print_section_header(title):
    print("\n==== {} ====".format(title))

def print_distribution(label, counter):
    total = sum(counter.values())
    print(f"{label} (n={total}):")
    if total == 0:
        print("  No cases found.")
    else:
        for k, v in counter.items():
            print(f"  {k}: {v}")

def print_agreement(label, agree_list):
    n = len(agree_list)
    n_agree = sum(agree_list)
    percent = 100 * n_agree / n if n else 0
    print(f"{label}: {n_agree}/{n} = {percent:.2f}%")

def analyse_llm_judge_agreement(use_soft_mapping=False):
    """
    Analyze agreement between annotations and LLM judge scores.
    For IDB Page 2: Compare annotation vs. narrative_attachment_type and linguistic_attachment_type
    """
    annotations = load_json(ANNOTATIONS_PATH)
    
    # Group by conversation_id and page
    by_convo = defaultdict(dict)
    for k, v in annotations.items():
        cid = v['conversation_id']
        page = v['page_num']
        by_convo[cid][page] = v

    # IDB Page 2 analysis - agreement with LLM judge
    narrative_agreement = []
    linguistic_agreement = []
    narrative_soft_agreement = []
    linguistic_soft_agreement = []
    
    for cid, pages in by_convo.items():
        if cid.startswith('idb') and 2 in pages:
            v2 = pages[2]
            style2 = v2['Demonstrated Attachment Style']
            
            # Find corresponding results file
            exp_id = get_exp_key(cid)
            found = False
            result = None
            
            for prefix in ["idb1_", "idb2_", "idb3_"]:
                fname = f"{prefix}{exp_id}.json"
                fpath = os.path.join(RESULTS_DIR, fname)
                if os.path.exists(fpath):
                    result = load_json(fpath)
                    found = True
                    break
            
            if found and result:
                # Agreement with narrative_attachment_type
                narrative_style = result.get('narrative_attachment_type', '')
                if narrative_style:
                    agree = (style2.lower() == narrative_style.lower())
                    narrative_agreement.append(agree)
                    if use_soft_mapping:
                        agree_soft = (soft_map(style2) == soft_map(narrative_style))
                        narrative_soft_agreement.append(agree_soft)
                
                # Agreement with linguistic_attachment_type
                linguistic_style = result.get('linguistic_attachment_type', '')
                if linguistic_style:
                    agree = (style2.lower() == linguistic_style.lower())
                    linguistic_agreement.append(agree)
                    if use_soft_mapping:
                        agree_soft = (soft_map(style2) == soft_map(linguistic_style))
                        linguistic_soft_agreement.append(agree_soft)

    print_section_header('IDB Page 2 - LLM Judge Agreement')
    print("ANALYSIS TYPE: LLM Judge Predictions vs. Human Annotations")
    print("PURPOSE: How well do LLM judge predictions agree with human annotations?")
    if not use_soft_mapping:
        print_agreement('Hard Agreement with Narrative Judge', narrative_agreement)
        print_agreement('Hard Agreement with Linguistic Judge', linguistic_agreement)
    else:
        print_agreement('Soft Agreement with Narrative Judge', narrative_soft_agreement)
        print_agreement('Soft Agreement with Linguistic Judge', linguistic_soft_agreement)
    print(f"Summary: Narrative comparisons: {len(narrative_agreement)}, Linguistic comparisons: {len(linguistic_agreement)}")
    print("==== End LLM Judge Agreement ====")

def enhanced_agreement_analysis(use_soft_mapping=False):
    """
    Enhanced agreement analysis with Kappa scores and per-class breakdowns
    """
    try:
        from sklearn.metrics import cohen_kappa_score, confusion_matrix
        from collections import Counter
    except ImportError:
        print("Error: scikit-learn not available. Install with: pip install scikit-learn")
        return
    
    annotations = load_json(ANNOTATIONS_PATH)
    
    # Group by conversation_id and page
    by_convo = defaultdict(dict)
    for k, v in annotations.items():
        cid = v['conversation_id']
        page = v['page_num']
        by_convo[cid][page] = v

    # Collect data for analysis
    narrative_pairs = []  # (annotation, prediction)
    linguistic_pairs = []
    
    for cid, pages in by_convo.items():
        if cid.startswith('idb') and 2 in pages:
            v2 = pages[2]
            style2 = v2['Demonstrated Attachment Style']
            
            # Find corresponding results file
            exp_id = get_exp_key(cid)
            found = False
            result = None
            
            for prefix in ["idb1_", "idb2_", "idb3_"]:
                fname = f"{prefix}{exp_id}.json"
                fpath = os.path.join(RESULTS_DIR, fname)
                if os.path.exists(fpath):
                    result = load_json(fpath)
                    found = True
                    break
            
            if found and result:
                # Process styles for comparison
                def process_style(s):
                    if use_soft_mapping:
                        return soft_map(s)
                    processed = s.lower() if s else 'undefined'
                    # Map 'none' to 'undefined' since they're semantically equivalent
                    if processed == 'none':
                        return 'undefined'
                    return processed
                
                ann_style = process_style(style2)
                
                # DEBUG: Print when we map 'none' to 'undefined'
                if style2 and style2.lower() == 'none':
                    print(f"DEBUG: Mapped annotation 'none' → 'undefined' for {cid}, original value: '{style2}'")
                
                # Narrative comparison
                narrative_style = result.get('narrative_attachment_type', '')
                if narrative_style:
                    pred_style = process_style(narrative_style)
                    if narrative_style.lower() == 'none':
                        print(f"DEBUG: Mapped narrative prediction 'none' → 'undefined' for {cid}, original value: '{narrative_style}'")
                    narrative_pairs.append((ann_style, pred_style))
                
                # Linguistic comparison
                linguistic_style = result.get('linguistic_attachment_type', '')
                if linguistic_style:
                    pred_style = process_style(linguistic_style)
                    if linguistic_style.lower() == 'none':
                        print(f"DEBUG: Mapped linguistic prediction 'none' → 'undefined' for {cid}, original value: '{linguistic_style}'")
                    linguistic_pairs.append((ann_style, pred_style))

    # Analysis function
    def analyze_pairs(pairs, judge_name):
        if not pairs:
            print(f"No data for {judge_name}")
            return
            
        annotations_list = [p[0] for p in pairs]
        predictions_list = [p[1] for p in pairs]
        
        # Calculate metrics
        kappa = cohen_kappa_score(annotations_list, predictions_list)
        raw_agreement = sum(1 for a, p in pairs if a == p) / len(pairs)
        
        print(f"\n=== {judge_name} Analysis ===")
        print(f"Total comparisons: {len(pairs)}")
        print(f"Raw Agreement: {raw_agreement:.3f} ({raw_agreement*100:.1f}%)")
        print(f"Cohen's Kappa: {kappa:.3f}")
        
        # Kappa interpretation
        if kappa < 0:
            kappa_interp = "Poor (worse than chance)"
        elif kappa < 0.20:
            kappa_interp = "Slight"
        elif kappa < 0.40:
            kappa_interp = "Fair"
        elif kappa < 0.60:
            kappa_interp = "Moderate"
        elif kappa < 0.80:
            kappa_interp = "Substantial"
        else:
            kappa_interp = "Almost Perfect"
        print(f"Kappa Interpretation: {kappa_interp}")
        
        # Per-class breakdown
        ann_counter = Counter(annotations_list)
        pred_counter = Counter(predictions_list)
        
        print(f"\nClass Distribution:")
        print(f"Annotations: {dict(ann_counter)}")
        print(f"Predictions: {dict(pred_counter)}")
        
        # Agreement by class
        print(f"\nPer-Class Agreement:")
        all_classes = set(annotations_list + predictions_list)
        for cls in sorted(all_classes):
            cls_pairs = [(a, p) for a, p in pairs if a == cls]
            if cls_pairs:
                cls_agreement = sum(1 for a, p in cls_pairs if a == p) / len(cls_pairs)
                print(f"  {cls}: {len(cls_pairs)} cases, {cls_agreement:.3f} agreement")
        
        # Confusion matrix
        print(f"\nConfusion Matrix (Rows=Annotation, Cols=Prediction):")
        classes = sorted(all_classes)
        cm = confusion_matrix(annotations_list, predictions_list, labels=classes)
        
        # Print header
        print("    " + "".join(f"{c:>8}" for c in classes))
        for i, true_class in enumerate(classes):
            print(f"{true_class:>3} " + "".join(f"{cm[i,j]:>8}" for j in range(len(classes))))
        
        # Major disagreements
        print(f"\nMajor Disagreements:")
        disagreements = Counter((a, p) for a, p in pairs if a != p)
        for (ann, pred), count in disagreements.most_common(5):
            print(f"  {ann} → {pred}: {count} cases")

    # Run analysis
    mapping_type = "Soft" if use_soft_mapping else "Hard"
    print_section_header(f'IDB Page 2 - Enhanced LLM Judge Agreement ({mapping_type})')
    print("ANALYSIS TYPE: LLM Judge Predictions vs. Human Annotations (Detailed)")
    print("PURPOSE: Detailed agreement analysis with Kappa scores and confusion matrices")
    
    analyze_pairs(narrative_pairs, "Narrative Judge")
    analyze_pairs(linguistic_pairs, "Linguistic Judge")
    
    print("==== End Enhanced Analysis ====")

def analyse_annotations(use_soft_mapping=False):
    annotations = load_json(ANNOTATIONS_PATH)
    experiment_mapping = load_json(EXPERIMENT_MAPPING_PATH)

    # Group by conversation_id and page
    by_convo = defaultdict(dict)
    for k, v in annotations.items():
        cid = v['conversation_id']
        page = v['page_num']
        by_convo[cid][page] = v

    # IAB analysis
    iab_styles = []
    iab_problematic = []
    for cid, pages in by_convo.items():
        if 'iab' in cid:
            v = pages.get(1)
            if v:
                style = v['Demonstrated Attachment Style']
                iab_styles.append(style)
                iab_problematic.append(v['Potentially Problematic'])
    print_section_header('IAB Cases')
    print_distribution('Attachment Style Distribution', Counter(iab_styles))
    print_distribution('Potentially Problematic Distribution', Counter(iab_problematic))
    print("Summary: IAB cases analyzed: {}".format(len(iab_styles)))
    print("==== End IAB ====")

    # IDB analysis
    idb1_styles, idb2_styles = [], []
    idb1_problematic, idb2_problematic = [], []
    idb1_agreement, idb2_agreement = [], []
    idb1_soft_agreement, idb2_soft_agreement = [], []
    for cid, pages in by_convo.items():
        if cid.startswith('idb'):
            # Page 1
            if 1 in pages:
                v1 = pages[1]
                style1 = v1['Demonstrated Attachment Style']
                idb1_styles.append(style1)
                idb1_problematic.append(v1['Potentially Problematic'])
                # Agreement with experiment mapping
                exp_key = get_exp_key(cid)
                exp = experiment_mapping.get(exp_key, {})
                exp_style = exp.get('attachment_type', None)
                if exp_style is not None:
                    agree = (style1.lower() == exp_style.lower())
                    idb1_agreement.append(agree)
                    if use_soft_mapping:
                        agree_soft = (soft_map(style1) == soft_map(exp_style))
                        idb1_soft_agreement.append(agree_soft)
            # Page 2
            if 2 in pages:
                v2 = pages[2]
                style2 = v2['Demonstrated Attachment Style']
                idb2_styles.append(style2)
                idb2_problematic.append(v2['Potentially Problematic'])
                # Agreement with page 1
                if 1 in pages:
                    style1 = pages[1]['Demonstrated Attachment Style']
                    agree = (style2.lower() == style1.lower())
                    idb2_agreement.append(agree)
                    if use_soft_mapping:
                        agree_soft = (soft_map(style2) == soft_map(style1))
                        idb2_soft_agreement.append(agree_soft)

    print_section_header('IDB Page 1 (First Page)')
    print("ANALYSIS TYPE: Human Annotation vs. Experimental Config")
    print("PURPOSE: Does the persona demonstrate the assigned experimental attachment type?")
    print_distribution('Attachment Style Distribution', Counter(idb1_styles))
    print_distribution('Potentially Problematic Distribution', Counter(idb1_problematic))
    if not use_soft_mapping:
        print_agreement('Hard Agreement with experiment mapping', idb1_agreement)
    else:
        print_agreement('Soft Agreement with experiment mapping', idb1_soft_agreement)
    print("Summary: IDB Page 1 cases analyzed: {}".format(len(idb1_styles)))
    print("==== End IDB Page 1 ====")

    print_section_header('IDB Page 2 (Second Page)')
    print("ANALYSIS TYPE: Human Annotation Page 2 vs. Human Annotation Page 1")
    print("PURPOSE: Inter-rater reliability between different pages of same conversation")
    print_distribution('Attachment Style Distribution', Counter(idb2_styles))
    print_distribution('Potentially Problematic Distribution', Counter(idb2_problematic))
    if not use_soft_mapping:
        print_agreement('Hard Agreement with Page 1', idb2_agreement)
    else:
        print_agreement('Soft Agreement with Page 1', idb2_soft_agreement)
    print("Summary: IDB Page 2 cases analyzed: {}".format(len(idb2_styles)))
    print("==== End IDB Page 2 ====")

if __name__ == '__main__':
    print('\n' + '='*40)
    print('ANNOTATION ANALYSIS REPORT')
    print('='*40)
    print('\n--- Hard Agreement Analysis ---')
    analyse_annotations(use_soft_mapping=False)
    analyse_llm_judge_agreement(use_soft_mapping=False)
    enhanced_agreement_analysis(use_soft_mapping=False)
    print('\n' + '='*40)
    print('--- Soft Agreement Analysis ---')
    print('='*40)
    analyse_annotations(use_soft_mapping=True)
    analyse_llm_judge_agreement(use_soft_mapping=True)
    enhanced_agreement_analysis(use_soft_mapping=True)
