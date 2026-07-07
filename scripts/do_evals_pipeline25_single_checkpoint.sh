#!/bin/bash

# === Arguments ===
MODEL_PATH=$1
BACKEND=$2
EVAL_DIR=$3
TOKENIZER_PATH=$4
RESULT_DIR=$5
SAVE_PREDICTIONS=$6
EVAL_PERPLEXITY_DIR=$7
shift 7
TASKS=("$@")  # All remaining arguments = task list

# === Handle default tasks if none are provided ===
if [ ${#TASKS[@]} -eq 0 ]; then
    TASKS=("perplexity" "reading" "blimp" "blimp_supp" "ewok" "entity_tracking" "wug")
fi

# === Flags & paths ===
SAVE_PREDICTIONS_FLAG=""
if [ "$SAVE_PREDICTIONS" = true ]; then
    SAVE_PREDICTIONS_FLAG="--save_predictions"
fi

echo "Starting evaluation of $MODEL_PATH with backend $BACKEND"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "Result directory: $RESULT_DIR"
echo "Tasks to run: ${TASKS[*]}"
echo "Evaluation directory: $EVAL_DIR"
echo "Save predictions: $SAVE_PREDICTIONS"

START_TIME=$(date +%s)

# === Environment setup ===
cd /cluster/home/janulm/thesis/evaluation-pipeline-2025
eval "$(micromamba shell hook --shell bash)"

# === Tokenizer copying ===
if [ ! -z "$TOKENIZER_PATH" ]; then
    echo "Copying tokenizer files from $TOKENIZER_PATH to $MODEL_PATH"
    cp -r "$TOKENIZER_PATH"/* "$MODEL_PATH"
fi

# === Run selected tasks ===
for task in "${TASKS[@]}"; do
    case "$task" in
        perplexity)
            micromamba activate llama
            echo "Running perplexity eval..."
            python /cluster/home/janulm/thesis/ms-thesis/evals/evaluate_checkpoint.py \
                --model_path "$MODEL_PATH" \
                --data_path "$EVAL_PERPLEXITY_DIR" \
                --output_dir "$RESULT_DIR" \
                $SAVE_PREDICTIONS_FLAG
            micromamba deactivate
            ;;

        reading)
            micromamba activate babylm-eval
            echo "Running reading eval..."
            python -m evaluation_pipeline.reading.run \
                --model_path_or_name "$MODEL_PATH" \
                --backend "$BACKEND" \
                --data_path "${EVAL_DIR}/reading/reading_data.csv" \
                --output_dir "$RESULT_DIR"
            micromamba deactivate
            ;;

        blimp)
            micromamba activate babylm-eval
            echo "Running BLiMP eval..."
            python -m evaluation_pipeline.sentence_zero_shot.run \
                --model_path_or_name "$MODEL_PATH" \
                --backend "$BACKEND" \
                --task blimp \
                --data_path "${EVAL_DIR}/blimp_filtered" \
                $SAVE_PREDICTIONS_FLAG \
                --output_dir "$RESULT_DIR"
            micromamba deactivate
            ;;

        blimp_supp)
            micromamba activate babylm-eval
            echo "Running BLiMP Supplement eval..."
            python -m evaluation_pipeline.sentence_zero_shot.run \
                --model_path_or_name "$MODEL_PATH" \
                --backend "$BACKEND" \
                --task blimp \
                --data_path "${EVAL_DIR}/supplement_filtered" \
                $SAVE_PREDICTIONS_FLAG \
                --output_dir "$RESULT_DIR"
            micromamba deactivate
            ;;

        ewok)
            micromamba activate babylm-eval
            echo "Running EWoK eval..."
            python -m evaluation_pipeline.sentence_zero_shot.run \
                --model_path_or_name "$MODEL_PATH" \
                --backend "$BACKEND" \
                --task ewok \
                --data_path "${EVAL_DIR}/ewok_filtered" \
                $SAVE_PREDICTIONS_FLAG \
                --output_dir "$RESULT_DIR"
            micromamba deactivate
            ;;

        entity_tracking)
            micromamba activate babylm-eval
            echo "Running Entity Tracking eval..."
            python -m evaluation_pipeline.sentence_zero_shot.run \
                --model_path_or_name "$MODEL_PATH" \
                --backend "$BACKEND" \
                --task entity_tracking \
                --data_path "${EVAL_DIR}/entity_tracking" \
                $SAVE_PREDICTIONS_FLAG \
                --output_dir "$RESULT_DIR"
            micromamba deactivate
            ;;

        wug)
            micromamba activate babylm-eval
            echo "Running WUG eval..."
            python -m evaluation_pipeline.sentence_zero_shot.run \
                --model_path_or_name "$MODEL_PATH" \
                --backend "$BACKEND" \
                --task wug \
                --data_path "${EVAL_DIR}/wug_adj_nominalization" \
                $SAVE_PREDICTIONS_FLAG \
                --output_dir "$RESULT_DIR"
            micromamba deactivate
            ;;

        *)
            echo "⚠ Unknown task: $task (skipping)"
            ;;
    esac
done

END_TIME=$(date +%s)
echo "Evaluation completed in $((END_TIME - START_TIME)) seconds"