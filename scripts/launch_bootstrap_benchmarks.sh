#!/bin/bash

# Usage: bash launch_bootstrap_benchmarks.sh /path/to/folder_with_model_folders

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/folder_with_model_folders"
    exit 1
fi

INPUT_PATH="$1"

if [ ! -d "$INPUT_PATH" ]; then
    echo "Error: $INPUT_PATH is not a directory."
    exit 1
fi

# Prepare micromamba for batch jobs
eval "$(micromamba shell hook --shell bash)"

# get the experiment name from the model path
EXPERIMENT_NAME=$(basename "$INPUT_PATH")
echo "Experiment name: $EXPERIMENT_NAME"

for MODEL_DIR in "$INPUT_PATH"/*/; do
    if [ -d "$MODEL_DIR" ]; then
        LOG_DIR="/cluster/home/janulm/thesis/ms-thesis/logs/${EXPERIMENT_NAME}"
        mkdir -p "$LOG_DIR"

        MODEL_NAME=$(basename "$MODEL_DIR")
        echo "Model name: $MODEL_NAME"

        JOB_NAME="bootbench_${EXPERIMENT_NAME}_${MODEL_NAME}"
        echo "Submitting job for $MODEL_DIR with name $JOB_NAME" 

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2096
#SBATCH --output=${LOG_DIR}/%j_%x.out
#SBATCH --error=${LOG_DIR}/%j_%x.err
#SBATCH --open-mode=append
#SBATCH --tmp=10000

# Activate micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate llama

# Run the python script
python /cluster/home/janulm/thesis/ms-thesis/evals/subsample_predictions.py --model_path "$MODEL_DIR"
EOF

    fi
done