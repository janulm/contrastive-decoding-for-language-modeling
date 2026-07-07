#!/bin/bash

EXPERIMENT_NAME="8000_series_all"
TOP_K=200
SYNTH_DATA_DIR="/cluster/work/cotterell/janulm/thesis_data/data/synthetic_data_generation_base8000/synth_no_contrast_topk_${TOP_K}/combined"
SYNTH_DATA_RATIO=0.3
MODEL_NAME="g2500_no_contrast_topk_${TOP_K}_mr03"
START_SEED=42
NUM_SEEDS=10

# Create logs directory for this experiment
mkdir -p "/cluster/home/janulm/thesis/ms-thesis/logs/${EXPERIMENT_NAME}"

# Function to submit a training job and return its job ID
submit_training_job() {
    local seed=$1
    local job_name="${EXPERIMENT_NAME}_${MODEL_NAME}_seed${seed}"
    
    # Create a temporary script for this training job
    TEMP_SCRIPT=$(mktemp)
    cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:4
#SBATCH --time=6:00:00
#SBATCH --job-name="${job_name}"
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2048
#SBATCH --tmp=10000
#SBATCH --output="/cluster/home/janulm/thesis/ms-thesis/logs/${EXPERIMENT_NAME}/%j_%x.out"
#SBATCH --error="/cluster/home/janulm/thesis/ms-thesis/logs/${EXPERIMENT_NAME}/%j_%x.err"
#SBATCH --open-mode=append

cd /cluster/home/janulm/thesis/ms-thesis

# Initialize micromamba for batch jobs
eval "\$(micromamba shell hook --shell bash)"
micromamba activate llama

# Print debug information
echo "=== Debug Information ==="
echo "SLURM_JOB_ID: \${SLURM_JOB_ID}"
echo "SLURM_NODEID: \${SLURM_NODEID}"
echo "SLURM_NODELIST: \${SLURM_NODELIST}"
echo "Current working directory: \$(pwd)"
echo "========================="

# Create output directory name
OUTPUT_DIR="/cluster/work/cotterell/janulm/thesis_data/checkpoints/models/${EXPERIMENT_NAME}/${MODEL_NAME}_${seed}"
CACHE_DATASET_DIR="/cluster/scratch/janulm/thesis/cache/${EXPERIMENT_NAME}/${MODEL_NAME}_${seed}"

# Create cache directory
mkdir -p "\${CACHE_DATASET_DIR}"

DATASET_DIR="/cluster/home/janulm/thesis/data/tinybabylm_dataset"
SYNTH_RATIO=${SYNTH_DATA_RATIO}
EVAL_STEPS=500
CHECKPOINT_STEPS=500
NUM_STEPS=8000
SEED=${seed}
LR=0.001

COMPILE_MODEL=true

if [ "\$COMPILE_MODEL" = true ]; then
    COMPILE_MODEL_ARG="--compile_model"
else
    COMPILE_MODEL_ARG=""
fi

# Calculate unique port for this job
MASTER_PORT=\$((29500 + \${SLURM_JOB_ID} % 1000))
MASTER_ADDR="localhost"

# Print port information
echo "=== Port Configuration ==="
echo "MASTER_PORT: \${MASTER_PORT}"
echo "MASTER_ADDR: \${MASTER_ADDR}"
echo "========================="

# Run the training script with 4 GPUs
echo "=== Starting torchrun ==="
torchrun --nproc_per_node=4 --master_port=\${MASTER_PORT} --master_addr=\${MASTER_ADDR} train_llama_steps.py \\
    --seed "\${SEED}" \\
    --learning_rate "\${LR}" \\
    \${COMPILE_MODEL_ARG} \\
    --output_dir "\${OUTPUT_DIR}" \\
    --dataset_dir "\${DATASET_DIR}" \\
    --lr_scheduler_type cosine \\
    --cache_dataset_dir "\${CACHE_DATASET_DIR}" \\
    --max_train_steps "\${NUM_STEPS}" \\
    --eval_steps "\${EVAL_STEPS}" \\
    --checkpointing_steps "\${CHECKPOINT_STEPS}" \\
    --synthetic_data_ratio "\${SYNTH_RATIO}" \\
    --synthetic_dataset_dir "${SYNTH_DATA_DIR}" \\
    --dataloader_num_workers 6 \\
    --model_name "${MODEL_NAME}_seed\${SEED}"

# Capture the exit status of torchrun
TRAINING_STATUS=\$?

echo "=== Training completed with status: \${TRAINING_STATUS} ==="

# Clean up cache directory
echo "Cleaning up cache directory..."
rm -rf "\${CACHE_DATASET_DIR}"

# Exit with the same status as the training script
exit \${TRAINING_STATUS}
EOF

    # Make the script executable
    chmod +x "$TEMP_SCRIPT"
    
    # Submit the job and capture its ID
    local job_id=$(sbatch --parsable "$TEMP_SCRIPT")
    
    # Clean up the temporary script
    rm "$TEMP_SCRIPT"
    
    echo $job_id
}

# Function to submit an evaluation job that depends on a training job
submit_eval_job() {
    local training_job_id=$1
    local seed=$2
    local job_name="${EXPERIMENT_NAME}_${MODEL_NAME}_launch_eval_seed${seed}"
    
    # Create a temporary script (this script will run eventually, but only after the training job has finished)
    TEMP_SCRIPT=$(mktemp)
    cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=0:15:00
#SBATCH --job-name="${job_name}"
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2048
#SBATCH --tmp=1000
#SBATCH --output="/cluster/home/janulm/thesis/ms-thesis/logs/${EXPERIMENT_NAME}/%j_%x.out"
#SBATCH --error="/cluster/home/janulm/thesis/ms-thesis/logs/${EXPERIMENT_NAME}/%j_%x.err"
#SBATCH --open-mode=append

set -e  # Exit on any error
set -x  # Print commands as they are executed

echo "Starting evaluation launch job for seed ${seed}"
echo "Current working directory: \$(pwd)"
echo "SLURM_JOB_ID: \${SLURM_JOB_ID}"

MODEL_PATH="/cluster/work/cotterell/janulm/thesis_data/checkpoints/models/${EXPERIMENT_NAME}/${MODEL_NAME}_${seed}"

# Wait for the model directory to exist and contain files
echo "Waiting for model directory to be ready..."
while [ ! -d "\${MODEL_PATH}" ] || [ -z "\$(ls -A \${MODEL_PATH})" ]; do
    echo "Model directory not ready yet, waiting..."
    sleep 30
done

echo "Model directory is ready: \${MODEL_PATH}"
ls -la "\${MODEL_PATH}"

# Run the evaluation pipeline script with the correct model path
echo "Launching evaluation pipeline..."
bash /cluster/home/janulm/thesis/ms-thesis/scripts/schedule_evals_pipeline25.sh "\${MODEL_PATH}"

echo "Evaluation pipeline launch completed"
EOF

    # Make the script executable
    chmod +x "$TEMP_SCRIPT"
    
    # Submit the job with dependency on the training job
    sbatch --dependency=afterok:${training_job_id} "$TEMP_SCRIPT"
    
    # Clean up the temporary script
    rm "$TEMP_SCRIPT"
}

# Launch training and evaluation jobs for each seed
for ((seed=START_SEED; seed<START_SEED+NUM_SEEDS; seed++)); do
    echo "Launching jobs for seed ${seed}"
    
    # Submit training job and get its ID
    training_job_id=$(submit_training_job $seed)
    echo "Submitted training job ${training_job_id} for seed ${seed}"
    
    # Submit evaluation job that depends on the training job
    submit_eval_job $training_job_id $seed
    echo "Submitted evaluation job for seed ${seed} (depends on job ${training_job_id})"
done

echo "All jobs submitted successfully!"