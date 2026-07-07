#!/bin/bash
set -euo pipefail

# ====== Experiment configuration ======
EXPERIMENT_NAME="synthetic_data_generation_base8000"

DATASET_NAME="synth_contrast_bad_500_n_gram"
CD_STRATEGY="contrastive_with_no_repeat_ngram"
TITLE="contrast-bad-500-ngram"
ALPHA=0.1
CONTRAST_STRENGTH=1.0
GOOD_MODEL_PATH="/cluster/work/cotterell/janulm/thesis_data/checkpoints/models/baseline_training/base_llama_8000_43/step_2500"
BAD_MODEL_PATH="/cluster/work/cotterell/janulm/thesis_data/checkpoints/models/baseline_training/base_llama_8000_43/step_500"

# Number of parallel 1-GPU jobs (== number of shards)
NUM_JOBS=20
ARRAY_MAX=$((NUM_JOBS - 1))

OUTPUT_DIR="/cluster/work/cotterell/janulm/thesis_data/data/${EXPERIMENT_NAME}/${DATASET_NAME}"
LOG_DIR="/cluster/home/janulm/thesis/ms-thesis/logs/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

# ====== Create generation script ======
GEN_SCRIPT="$(mktemp)"
cat > "${GEN_SCRIPT}" << 'EOF'
#!/bin/bash
#SBATCH --job-name=sdg_gen
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2048
#SBATCH --tmp=10000
#SBATCH --output=__LOG_DIR__/%A_%a_%x.out
#SBATCH --error=__LOG_DIR__/%A_%a_%x.err
#SBATCH --open-mode=append

set -euo pipefail

cd /cluster/home/janulm/thesis/ms-thesis

# Init micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate llama

echo "=== Gen Task Debug ==="
echo "JOB_ID: ${SLURM_JOB_ID}  ARRAY_ID: ${SLURM_ARRAY_JOB_ID}  TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "PWD : $(pwd)"
echo "======================="

COMPILE_MODEL=false
if [ "${COMPILE_MODEL}" = true ]; then
  COMPILE_FLAG="--compile_model"
else
  COMPILE_FLAG=""
fi

BATCH_SIZE=16

python generate_synthetic_data_better.py \
  --output_dataset_path "__OUTPUT_DIR__" \
  --seed_dataset_path "/cluster/home/janulm/thesis/data/tinybabylm_dataset/seed" \
  --cd_strategy "__CD_STRATEGY__" \
  --title "__TITLE__" \
  --good_model_path "__GOOD_MODEL_PATH__" \
  --bad_model_path "__BAD_MODEL_PATH__" \
  --contrast_strength "__CONTRAST_STRENGTH__" \
  ${COMPILE_FLAG} \
  --batch_size "${BATCH_SIZE}" \
  --alpha "__ALPHA__" \
  --num_workers "__NUM_JOBS__" \
  --worker_id "${SLURM_ARRAY_TASK_ID}"
EOF

# Fill placeholders with literals
sed -i \
  -e "s|__LOG_DIR__|${LOG_DIR}|g" \
  -e "s|__OUTPUT_DIR__|${OUTPUT_DIR}|g" \
  -e "s|__CD_STRATEGY__|${CD_STRATEGY}|g" \
  -e "s|__TITLE__|${TITLE}|g" \
  -e "s|__GOOD_MODEL_PATH__|${GOOD_MODEL_PATH}|g" \
  -e "s|__BAD_MODEL_PATH__|${BAD_MODEL_PATH}|g" \
  -e "s|__CONTRAST_STRENGTH__|${CONTRAST_STRENGTH}|g" \
  -e "s|__ALPHA__|${ALPHA}|g" \
  -e "s|__NUM_JOBS__|${NUM_JOBS}|g" \
  "${GEN_SCRIPT}"

chmod +x "${GEN_SCRIPT}"

# Submit job array with literal range
gen_array_id=$(sbatch --parsable --array=0-${ARRAY_MAX} "${GEN_SCRIPT}")
echo "Submitted generation array: ${gen_array_id}"

# ====== Create combine script ======
COMB_SCRIPT="$(mktemp)"
cat > "${COMB_SCRIPT}" << 'EOF'
#!/bin/bash
#SBATCH --job-name=sdg_combine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:30:00
#SBATCH --mem-per-cpu=6048
#SBATCH --output=__LOG_DIR__/%j_%x.out
#SBATCH --error=__LOG_DIR__/%j_%x.err
#SBATCH --open-mode=append

set -euo pipefail

cd /cluster/home/janulm/thesis/ms-thesis

# Init micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate llama

echo "=== Combining shards from __OUTPUT_DIR__ ==="
python combine_synthetic_datasets_better.py \
  --input_dir "__OUTPUT_DIR__" \
  --output_dir "__OUTPUT_DIR__/combined" \
  --cleanup

# Optional: launch training afterwards
if [[ "${LAUNCH_TRAINING:-false}" == "true" ]]; then
  echo "Launching training..."
  bash "/cluster/home/janulm/thesis/ms-thesis/scripts/experiments/3_8000series_all/g2500_500_n_gram_cs1_mr03.sh"
fi
EOF

sed -i \
  -e "s|__LOG_DIR__|${LOG_DIR}|g" \
  -e "s|__OUTPUT_DIR__|${OUTPUT_DIR}|g" \
  "${COMB_SCRIPT}"

chmod +x "${COMB_SCRIPT}"

# Submit combine job with dependency on the array
comb_id=$(sbatch --parsable --dependency=afterok:${gen_array_id} "${COMB_SCRIPT}")
echo "Submitted combine job (after all shards OK): ${comb_id}"

# Cleanup temp scripts
rm -f "${GEN_SCRIPT}" "${COMB_SCRIPT}"

echo "All jobs submitted successfully."