#!/bin/bash

# Arrays of models and experiment types
MODELS=( "mistral" "gpt-cheap" "gemini"  "llama" "deepseek"   )
EXPERIMENTS=( "iab" "idb1" "idb2"  )

# Ensure results directory exists
mkdir -p results

# Function to run experiment
run_experiment() {
    local primary=$1
    local exp_type=$2
    
    echo "Running ${exp_type} evaluation with ${primary}"
    python src/llm_attachment_index/experiment.py \
        --run "${exp_type}" \
        --primary "${primary}" \
        --strong_priming
}

# Run all experiments for all models
for exp_type in "${EXPERIMENTS[@]}"; do
    echo "=== Running ${exp_type} Evaluations ==="
    for model in "${MODELS[@]}"; do
        run_experiment "${model}" "${exp_type}"
    done
done

echo "All evaluations completed!" 
