#!/bin/bash

# Expect user to pass something like:
# /data/araghavan/HIATUS/datadreamer-ta2/src/run_eval_script_on_lora_training.sh sadiri_hardpos_rankingloss_random_hardbatching_v001/20250402_193508
# Check if argument was passed
if [ -z "$1" ]; then
  echo "Usage: $0 <BASE_DIR_NAME>"
  echo "Example: $0 sadiri_hardpos_rankingloss_random_hardbatching_v001/20250402_193508"
  exit 1
fi

BASE_DIR_NAME="${1}"

# Remove slashes → underscores
BASE_DIR_NAME_NOSLASH=$(echo "$BASE_DIR_NAME" | tr '/' '_')

# Paths
MODEL_CHECKPOINT="/data/araghavan/HIATUS/datadreamer-ta2/output/${BASE_DIR_NAME}/luar-trainer/_model"
PAUSIT_EVAL_TA2_DIR="/data/araghavan/HIATUS/pausit-eval-ta2"
LORA_CHECKPOINT="${PAUSIT_EVAL_TA2_DIR}/SIVs/datadreamer_lora/checkpoint"
BASE_INPUT_DIR="/data/araghavan/HIATUS/datadreamer-ta2/data/hrs/nikita_direct_splits_5_random_fold_splits_qc/crossGenre"

# Derived paths
BASE_OUTPUT_DIR="/data/araghavan/HIATUS/datadreamer-ta2/data/hrs/nikita_direct_splits_5_random_fold_splits_qc_outputs/kft_${BASE_DIR_NAME}"
LOGFILE="/data/araghavan/HIATUS/datadreamer-ta2/output/logs/kft_${BASE_DIR_NAME_NOSLASH}_eval_op.txt"

# Check if model checkpoint exists
if [ ! -d "$MODEL_CHECKPOINT" ]; then
  echo "Model checkpoint not found at: $MODEL_CHECKPOINT"
  exit 1
fi

echo "Copying Model Checkpoint to LoRA Checkpoint Directory"
echo "Running Command: cp -r $MODEL_CHECKPOINT/. $LORA_CHECKPOINT"
cp -r "${MODEL_CHECKPOINT}/." "$LORA_CHECKPOINT"

# Activate conda env `hiatus`
eval "$(conda shell.bash hook)"
conda activate hiatus

# Constants
RUN_ID_START=5


echo "==== K-Fold Run Started at $(date) ===="
echo "Writing to log file: ######## $LOGFILE ########"

echo "==== K-Fold Run Started at $(date) ====" > "$LOGFILE"
echo "Using BASE_OUTPUT_DIR: $BASE_OUTPUT_DIR" >> "$LOGFILE"
cd "${PAUSIT_EVAL_TA2_DIR}" || exit 1

for fold in {1..5}; do
  # K Fold Data Dirs
  INPUT_DIR="${BASE_INPUT_DIR}/fold_${fold}/test/TA2/hrs2_09-24-24_english_crossGenre-combined/data/"
  GROUNDTRUTH_DIR="${BASE_INPUT_DIR}/fold_${fold}/test/TA2/hrs2_09-24-24_english_crossGenre-combined/groundtruth/"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/0${fold}"
  RUN_ID=$((RUN_ID_START + fold - 1))

  echo "==== Running Fold ${fold} (RUN_ID=${RUN_ID}) ===="
  echo "==== Running Fold ${fold} (RUN_ID=${RUN_ID}) ====" >> "$LOGFILE"
  echo "Running Command:" >> "$LOGFILE"
  echo "python -u main.py --input-dir \"$INPUT_DIR\" --output-dir \"$OUTPUT_DIR\" --ground-truth-dir \"$GROUNDTRUTH_DIR\" --run-id \"$RUN_ID\" --query-identifier authorIDs --candidate-identifier authorSetIDs -ta1 datadreamer_lora -g -ta2 baseline --language eng" >> "$LOGFILE"

  python -u main.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --ground-truth-dir "$GROUNDTRUTH_DIR" \
    --run-id "$RUN_ID" \
    --query-identifier authorIDs \
    --candidate-identifier authorSetIDs \
    -ta1 datadreamer_lora \
    -g \
    -ta2 baseline \
    --language eng >> "$LOGFILE" 2>&1 
    
    if [ $? -eq 0 ]; then
      echo -e "\t\t==== Fold ${fold} Completed ====\n" | tee -a "$LOGFILE"
    else
      echo -e "\t\t❌ Fold ${fold} FAILED!\n" | tee -a "$LOGFILE"
    fi
done

echo "==== K-Fold Run Finished at $(date) ====" >> "$LOGFILE"
echo "Final Output Directory: $BASE_OUTPUT_DIR" >> "$LOGFILE"

# Check that all 5 folds completed successfully
FOLD_COMPLETE_COUNT=$(grep -c "==== Fold [1-5] Completed ====" "$LOGFILE")

if [ "$FOLD_COMPLETE_COUNT" -eq 5 ]; then
  echo "" >> "$LOGFILE"
  echo "==== All 5 folds completed. Calculating average metrics... ====" | tee -a "$LOGFILE"

  # Initialize accumulators
  declare -A sum
  declare -a metrics=(
    "Area Under ROC Curve"
    "partial Area Under ROC Curve"
    "Precision"
    "Recall"
    "F1"
    "Equal Error Rate"
    "Detection Cost Function"
    "TAR@0.05"
  )

  for metric in "${metrics[@]}"; do
    # Extract values, strip leading/trailing whitespace, and sum
    values=$(grep "^$metric" "$LOGFILE" | awk '{print $NF}')
    total=0
    count=0

    for val in $values; do
      total=$(echo "$total + $val" | bc -l)
      ((count++))
    done

    if [ "$count" -gt 0 ]; then
      avg=$(echo "scale=6; $total / $count" | bc -l)
      sum["$metric"]=$avg
    else
      sum["$metric"]="N/A"
    fi
  done

  echo "==== Averaged Metrics Over 5 Folds ====" | tee -a "$LOGFILE"
  for metric in "${metrics[@]}"; do
    printf "%-30s %s\n" "$metric" "${sum[$metric]}" | tee -a "$LOGFILE"
  done
  echo "=======================================" | tee -a "$LOGFILE"

else
  echo "❌ Not all folds completed. Skipping metrics summary." | tee -a "$LOGFILE"
fi

echo "Log File: $LOGFILE"
