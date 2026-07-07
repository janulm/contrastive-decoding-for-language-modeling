#!/bin/bash

BASE_EVAL_DIR="/cluster/work/cotterell/janulm/thesis_data/eval_results"
BASE_MODEL_DIR="/cluster/work/cotterell/janulm/thesis_data/checkpoints/models"
SCRIPT_TO_CALL="/cluster/home/janulm/thesis/ms-thesis/scripts/do_evals_pipeline25_single_checkpoint.sh"

VERBOSE=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -v|--verbose) VERBOSE=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Map output files to eval tasks
declare -A file_to_task=(
    ["tinybabylm_dataset-test/per_batch_losses.npy"]="perplexity"
    ["tinybabylm_dataset-test/best_temperature_report.txt"]="perplexity"
    ["main/zero_shot/causal/reading/prediction.jsonl"]="reading"
    ["main/zero_shot/causal/blimp/blimp_filtered/best_temperature_report.txt"]="blimp"
    ["main/zero_shot/causal/blimp/supplement_filtered/best_temperature_report.txt"]="blimp_supp"
    ["main/zero_shot/causal/ewok/ewok_filtered/best_temperature_report.txt"]="ewok"
    ["main/zero_shot/causal/wug/wug_adj_nominalization/best_temperature_report.txt"]="wug"
    ["main/zero_shot/causal/entity_tracking/entity_tracking/best_temperature_report.txt"]="entity_tracking"
)

# Constants
BACKEND="causal"
EVAL_DIR="/cluster/home/janulm/thesis/evaluation-pipeline-2025/evaluation_data/full_eval"
EVAL_PERPLEXITY_DIR="/cluster/home/janulm/thesis/data/tinybabylm_dataset/test"
TOKENIZER_PATH="/cluster/home/janulm/thesis/checkpoints/tokenizers/tinybabylm_100_32"

# Scan models
for model_result_path in "$BASE_EVAL_DIR"/*/*; do
    [[ -d "$model_result_path" ]] || continue

    echo "Processing model: $model_result_path"
    
    model_name=$(basename "$model_result_path")
    experiment_group=$(basename "$(dirname "$model_result_path")")
    experiment_name="${experiment_group}/${model_name}"
    model_checkpoint_base="${BASE_MODEL_DIR}/${experiment_name}"

    for step_dir in "$model_result_path"/step_*; do
        [[ -d "$step_dir" ]] || continue
        step_name=$(basename "$step_dir")
        checkpoint_path="${model_checkpoint_base}/${step_name}/"
        result_path="${model_result_path}"

        declare -A task_missing=()

        for rel_path in "${!file_to_task[@]}"; do
            task="${file_to_task[$rel_path]}"
            full_path="${step_dir}/${rel_path}"
            if [[ ! -f "$full_path" ]]; then
                task_missing["$task"]=1
                $VERBOSE && echo "✘ Missing $full_path (→ task $task)"
                #else
                #$VERBOSE && echo "✔ Found: $full_path"
            fi
        done

        if [ ${#task_missing[@]} -gt 0 ]; then
            # Create task list
            task_list=("${!task_missing[@]}")
            task_str="${task_list[*]}"
            echo "⚙ Launching eval for checkpoint $checkpoint_path (tasks: $task_str)"

            TEMP_SCRIPT=$(mktemp)
            cat > "$TEMP_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=1:00:00
#SBATCH --job-name="eval-${model_name}-${step_name}"
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=10000
#SBATCH --output="/cluster/home/janulm/thesis/ms-thesis/logs/${experiment_name}/%j_%x.out"
#SBATCH --error="/cluster/home/janulm/thesis/ms-thesis/logs/${experiment_name}/%j_%x.err"
#SBATCH --open-mode=append

bash "$SCRIPT_TO_CALL" \\
  "$checkpoint_path" \\
  "$BACKEND" \\
  "$EVAL_DIR" \\
  "$TOKENIZER_PATH" \\
  "$result_path" \\
  true \\
  "$EVAL_PERPLEXITY_DIR" \\
  ${task_list[@]}
EOF

            chmod +x "$TEMP_SCRIPT"
            sbatch "$TEMP_SCRIPT"
            rm "$TEMP_SCRIPT"
        fi
    done
done