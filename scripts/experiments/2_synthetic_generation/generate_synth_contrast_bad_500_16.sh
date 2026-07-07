#!/bin/bash

EXPERIMENT_NAME="synthetic_data_generation_base8000"

DATASET_NAME="synth_contrast_bad_500_16"
CD_STRATEGY="contrastive"
TITLE="contrast-bad-500-16"
ALPHA=0.1
CONTRAST_STRENGTH=1.0
GOOD_MODEL_PATH="/cluster/work/cotterell/janulm/thesis_data/checkpoints/models/baseline_training/base_llama_8000_43/step_2500"
BAD_MODEL_PATH="/cluster/work/cotterell/janulm/thesis_data/checkpoints/models/baseline_training/base_llama_8000_43/step_500"
NUMB_GPUS=4

OUTPUT_DIR="/cluster/work/cotterell/janulm/thesis_data/data/${EXPERIMENT_NAME}/${DATASET_NAME}"

# Create logs directory for this experiment
mkdir -p "/cluster/home/janulm/thesis/ms-thesis/logs/${EXPERIMENT_NAME}"


# Create a temporary script for this generation job
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus=${NUMB_GPUS}
#SBATCH --gres=gpumem:24g
#SBATCH --time=5:00:00
#SBATCH --job-name="${EXPERIMENT_NAME}_${DATASET_NAME}"
#SBATCH --cpus-per-task=48
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


COMPILE_MODEL=false

if [ "\$COMPILE_MODEL" = true ]; then
    COMPILE_MODEL_ARG="--compile_model"
else
    COMPILE_MODEL_ARG=""
fi

BATCH_SIZE=16

# Calculate unique port for this job
MASTER_PORT=\$((29500 + \${SLURM_JOB_ID} % 1000))
MASTER_ADDR="localhost"

# Print port information
echo "=== Port Configuration ==="
echo "MASTER_PORT: \${MASTER_PORT}"
echo "MASTER_ADDR: \${MASTER_ADDR}"
echo "========================="

# Run the generation script

# Run the training script with 8 GPUs
echo "=== Starting torchrun ==="
torchrun --nproc_per_node=${NUMB_GPUS} --master_port=\${MASTER_PORT} --master_addr=\${MASTER_ADDR} generate_synthetic_data.py \\
    --output_dataset_path "${OUTPUT_DIR}" \\
    --seed_dataset_path "/cluster/home/janulm/thesis/data/tinybabylm_dataset/seed" \\
    --cd_strategy "${CD_STRATEGY}" \\
    --title "${TITLE}" \\
    --good_model_path "${GOOD_MODEL_PATH}" \\
    --bad_model_path "${BAD_MODEL_PATH}" \\
    --contrast_strength "${CONTRAST_STRENGTH}" \\
    \${COMPILE_MODEL_ARG} \\
    --batch_size "\${BATCH_SIZE}" \\
    --numb_generations_per_seed 16 \\
    --alpha "${ALPHA}"

# Capture the exit status
GENERATION_STATUS=\$?

echo "=== Generation completed with status: \${GENERATION_STATUS} ==="

# If generation was successful, combine the datasets
if [ \${GENERATION_STATUS} -eq 0 ]; then
    echo "=== Starting dataset combination ==="
    python combine_synthetic_datasets.py \\
        --input_dir "${OUTPUT_DIR}" \\
        --output_dir "${OUTPUT_DIR}/combined" \\
        --cleanup
    COMBINE_STATUS=\$?
    echo "=== Dataset combination completed with status: \${COMBINE_STATUS} ==="
    exit \${COMBINE_STATUS}
else
    exit \${GENERATION_STATUS}
fi
EOF

# Make the script executable
chmod +x "$TEMP_SCRIPT"

# Submit the job
job_id=$(sbatch --parsable "$TEMP_SCRIPT")
echo "Submitted generation job ${job_id}"

# Clean up the temporary script
rm "$TEMP_SCRIPT"

echo "Job submitted successfully!"
