#!/bin/bash

# Base directory for pipeline25 experiments
PIPELINE_DIR="/cluster/home/janulm/thesis/ms-thesis/evals/eval_jobs_results/pipeline25"

# Initialize micromamba properly
eval "$(micromamba shell hook --shell bash)"
micromamba activate llama

# Check if pipeline25 directory exists
if [ ! -d "$PIPELINE_DIR" ]; then
    echo "Error: $PIPELINE_DIR directory not found"
    exit 1
fi

# Process each experiment folder
for exp_dir in "$PIPELINE_DIR"/*/; do
    if [ ! -d "$exp_dir" ]; then
        continue
    fi
    
    # Get experiment name from directory name and remove trailing slash
    exp_dir="${exp_dir%/}"
    exp_name=$(basename "$exp_dir")
    echo "Processing experiment: $exp_name"
    
    # Create plots directory
    plots_dir="${exp_dir}/plots"
    mkdir -p "$plots_dir"
    
    # Extract data to DataFrame and create seed analysis
    echo "Extracting data to DataFrame and creating seed analysis..."
    python /cluster/home/janulm/thesis/ms-thesis/evals/extract_experiment_dataframe.py \
        "$exp_dir" \
        --output "${plots_dir}/${exp_name}_data.csv" \
        --output-seed-analysis "${plots_dir}/${exp_name}_seed_analysis.csv"
    
    # Create plots with seed analysis
    echo "Creating plots with seed analysis..."
    python /cluster/home/janulm/thesis/ms-thesis/evals/plot_experiment_seeds.py \
        "$exp_dir" \
        "$plots_dir" \
        --figsize 12 8 \
        --dpi 1200 \
        --export-latex-table
    
    echo "Completed processing experiment: $exp_name"
    echo "----------------------------------------"
done

echo "All experiments processed"
