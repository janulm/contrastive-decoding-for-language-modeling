#!/bin/bash

BASE_EVAL_DIR="/cluster/work/cotterell/janulm/thesis_data/eval_results"
BASE_MODEL_DIR="/cluster/work/cotterell/janulm/thesis_data/checkpoints/models"
SCRIPT_TO_CALL="/cluster/home/janulm/thesis/ms-thesis/evals/subsample_predictions_single.py"
INDEX_SET_PATH="/cluster/work/cotterell/janulm/thesis_data/index_sets/"

VERBOSE=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -v|--verbose) VERBOSE=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Map output files to bootstrap tasks
declare -A file_to_task=(
    ["main/zero_shot/causal/blimp/blimp_filtered/subsampled_scores.npy"]="blimp"
    ["main/zero_shot/causal/blimp/supplement_filtered/subsampled_scores.npy"]="blimp_supplement"
    ["main/zero_shot/causal/entity_tracking/entity_tracking/subsampled_scores.npy"]="entity_tracking"
    ["main/zero_shot/causal/ewok/ewok_filtered/subsampled_scores.npy"]="ewok"
    ["main/zero_shot/causal/wug/wug_adj_nominalization/subsampled_scores.npy"]="wug"
    ["tinybabylm_dataset-test/subsampled_scores.npy"]="perplexity"
    ["main/zero_shot/causal/reading/subsampled_scores_eye_tracking.npy"]="eye_tracking"
    ["main/zero_shot/causal/reading/subsampled_scores_reading.npy"]="reading"
)

GROUND_TRUTH_PATH="/cluster/home/janulm/thesis/evaluation-pipeline-2025/evaluation_data/full_eval/"
SUBSAMPLE_SIZE=1000

# Scan models
for model_result_path in "$BASE_EVAL_DIR"/*/*; do
    [[ -d "$model_result_path" ]] || continue

    echo "Processing model: $model_result_path"

    model_name=$(basename "$model_result_path")
    experiment_group=$(basename "$(dirname "$model_result_path")")
    experiment_name="${experiment_group}/${model_name}"
    model_checkpoint_base="${BASE_EVAL_DIR}/${experiment_name}"

    for step_dir in "$model_result_path"/step_*; do
        [[ -d "$step_dir" ]] || continue
        step_name=$(basename "$step_dir")
        checkpoint_path="${model_checkpoint_base}/${step_name}/"

        declare -A task_missing=()

        for rel_path in "${!file_to_task[@]}"; do
            task="${file_to_task[$rel_path]}"
            full_path="${step_dir}/${rel_path}"
            if [[ ! -f "$full_path" ]]; then
                task_missing["$task"]=1
                $VERBOSE && echo "✘ Missing $full_path (→ task $task)"
            fi
        done

        if [ ${#task_missing[@]} -gt 0 ]; then
            task_list=("${!task_missing[@]}")
            task_str="${task_list[*]}"
            echo "⚙ Launching bootstrap for checkpoint $checkpoint_path (tasks: $task_str)"

            TEMP_SCRIPT=$(mktemp)
            cat > "$TEMP_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=10000
#SBATCH --job-name="boot-${model_name}-${step_name}"
#SBATCH --output="/cluster/home/janulm/thesis/ms-thesis/logs/${experiment_name}/%j_%x.out"
#SBATCH --error="/cluster/home/janulm/thesis/ms-thesis/logs/${experiment_name}/%j_%x.err"
#SBATCH --open-mode=append

eval "\$(micromamba shell hook --shell bash)"
micromamba activate llama

python "$SCRIPT_TO_CALL" \
  --model_path "$checkpoint_path" \
  --index_set_path "$INDEX_SET_PATH" \
  --ground_truth_path "$GROUND_TRUTH_PATH" \
  --tasks ${task_list[@]}
EOF

            chmod +x "$TEMP_SCRIPT"
            sbatch "$TEMP_SCRIPT"
            rm "$TEMP_SCRIPT"
        fi
    done
done