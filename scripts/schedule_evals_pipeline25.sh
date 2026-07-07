#!/bin/bash

# Check if model path is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

# Get the model path from command line argument
MODEL_PATH="$1"

# Flag to control whether to save predictions (true/false)
SAVE_PREDICTIONS=true

START_TIME=$(date +%s)

# get the experiment name from the model path
EXPERIMENT_NAME=$(basename $(dirname "$MODEL_PATH"))
echo "Experiment name: $EXPERIMENT_NAME"

## SHOULDNT CHANGE THE PARAMETERS BELOW
BACKEND="causal"

# this is the data for the 2025 evaluation pipeline and the correspoding tasks
EVAL_DIR="/cluster/home/janulm/thesis/evaluation-pipeline-2025/evaluation_data/full_eval"

# this is the data for our own custom dataset (real data)
EVAL_PERPLEXITY_DIR="/cluster/home/janulm/thesis/data/tinybabylm_dataset/test"

TOKENIZER_PATH="/cluster/home/janulm/thesis/checkpoints/tokenizers/tinybabylm_100_32"
RESULT_DIR="/cluster/work/cotterell/janulm/thesis_data/eval_results/${EXPERIMENT_NAME}"



# Extract model name (last directory in path)
MODEL_NAME=$(basename "$MODEL_PATH")
echo "Model name: $MODEL_NAME"

# List all subdirectories in the model path
echo "Subdirectories in model path:"
ls -d "$MODEL_PATH"/*/

# Create a temporary script for each checkpoint
for checkpoint_dir in "$MODEL_PATH"/*/; do
    # Get the checkpoint name (basename of the directory)
    checkpoint_name=$(basename "$checkpoint_dir")
    
    # Create the new result directory path
    checkpoint_result_dir="${RESULT_DIR}/${MODEL_NAME}"
    
    echo "Submitting job for checkpoint: $checkpoint_name"
    
    # Create a temporary script for this checkpoint
    TEMP_SCRIPT=$(mktemp)
    cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=1:00:00
#SBATCH --job-name="eval-${EXPERIMENT_NAME}-${MODEL_NAME}-${checkpoint_name}"
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=10000
#SBATCH --output="/cluster/home/janulm/thesis/ms-thesis/logs/${EXPERIMENT_NAME}/%j_%x.out"
#SBATCH --error="/cluster/home/janulm/thesis/ms-thesis/logs/${EXPERIMENT_NAME}/%j_%x.err"
#SBATCH --open-mode=append

bash /cluster/home/janulm/thesis/ms-thesis/scripts/do_evals_pipeline25_single_checkpoint.sh \\
    "$checkpoint_dir" \\
    "$BACKEND" \\
    "$EVAL_DIR" \\
    "$TOKENIZER_PATH" \\
    "$checkpoint_result_dir" \\
    "$SAVE_PREDICTIONS" \\
    "$EVAL_PERPLEXITY_DIR"
EOF

    # Make the script executable
    chmod +x "$TEMP_SCRIPT"
    
    # Submit the job
    sbatch "$TEMP_SCRIPT"
    
    # Clean up the temporary script
    rm "$TEMP_SCRIPT"
done

END_TIME=$(date +%s)
echo "All jobs submitted in $((END_TIME - START_TIME)) seconds"
