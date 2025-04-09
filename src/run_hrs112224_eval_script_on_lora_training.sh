#!/bin/bash

# Check for input
if [ -z "$1" ]; then
  echo "Usage: $0 <BASE_DIR_NAME>"
  echo "Example: $0 sadiri_hardpos_rankingloss_random_hardbatching_v001/20250402_193508"
  exit 1
fi

BASE_DIR_NAME="${1}"
BASE_DIR_NAME_NOSLASH=$(echo "$BASE_DIR_NAME" | tr '/' '_')

MODEL_CHECKPOINT="/data/araghavan/HIATUS/datadreamer-ta2/output/${BASE_DIR_NAME}/luar-trainer/_model"
PAUSIT_EVAL_TA2_DIR="/data/araghavan/HIATUS/pausit-eval-ta2"
LORA_CHECKPOINT="${PAUSIT_EVAL_TA2_DIR}/SIVs/datadreamer_lora/checkpoint"
LOGFILE="/data/araghavan/HIATUS/datadreamer-ta2/output/logs/hrs_${BASE_DIR_NAME_NOSLASH}_eval_op.txt"

# Input paths
declare -A INPUT_PATHS=(
  ["hrs1.1"]="/data/araghavan/HIATUS/datadreamer-ta2/data/hrs/hrs_release_11-22-24/HRS_evaluation_samples/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.1"
  ["hrs1.2"]="/data/araghavan/HIATUS/datadreamer-ta2/data/hrs/hrs_release_11-22-24/HRS_evaluation_samples/HRS1_english_long/TA2/HRS1_english_long_sample-0_perGenre-HRS1.2"
  ["hrs2.1"]="/data/araghavan/HIATUS/datadreamer-ta2/data/hrs/hrs_release_11-22-24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.1"
  ["hrs2.2"]="/data/araghavan/HIATUS/datadreamer-ta2/data/hrs/hrs_release_11-22-24/HRS_evaluation_samples/HRS2_english_medium/TA2/HRS2_english_medium_sample-0_perGenre-HRS2.2"
)
HRS_TYPES=("hrs1.1" "hrs1.2" "hrs2.1" "hrs2.2")

# Check model checkpoint
if [ ! -d "$MODEL_CHECKPOINT" ]; then
  echo "❌ Model checkpoint not found at: $MODEL_CHECKPOINT"
  exit 1
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate hiatus

echo "==== HRS Evaluation Started at $(date) ===="
echo "Log File: $LOGFILE"

echo "==== HRS Evaluation Started at $(date) ====" > "$LOGFILE"

echo "Running Command: cp -r $MODEL_CHECKPOINT/. $LORA_CHECKPOINT"
cp -r "${MODEL_CHECKPOINT}/." "$LORA_CHECKPOINT"

cd "${PAUSIT_EVAL_TA2_DIR}" || exit 1
RUN_ID=14

for HRS in "${HRS_TYPES[@]}"; do
  INPUT_DIR="${INPUT_PATHS[$HRS]}/data"
  GROUNDTRUTH_DIR="${INPUT_PATHS[$HRS]}/groundtruth"
  OUTPUT_DIR="/data/araghavan/HIATUS/datadreamer-ta2/data/hrs/hrs_release_11-22-24_outputs/${HRS}"

  echo "==== Running ${HRS} (RUN_ID=${RUN_ID}) ====" | tee -a "$LOGFILE"
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

  exit_code=${PIPESTATUS[0]}

  if [ $exit_code -eq 0 ]; then
    echo -e "\t✅ ${HRS} Completed\n"
    echo "==== ${HRS} Completed ====" >> "$LOGFILE"
  else
    echo -e "\t❌ ${HRS} FAILED (exit code $exit_code)"
    echo "==== ${HRS} FAILED ==== Exiting ==== " >> "$LOGFILE"
    exit 1 
  fi

  grep -B 15 "==== ${HRS} Completed ====" "${LOGFILE}" | sed -n '/^Area Under ROC Curve/,/TAR@0.05/p'

  ((RUN_ID++))
done

# Print averages
echo "" >> "$LOGFILE"
echo "==== Averaged Metrics Across All HRS ====" | tee -a "$LOGFILE"
# Define the metrics (update as needed)
metrics=(
  "Area Under ROC Curve"
  "partial Area Under ROC Curve"
  "Precision"
  "Recall"
  "F1"
  "Equal Error Rate"
  "Detection Cost Function"
  "TAR@0.05"
)

# Loop over each metric and calculate its average value from the logfile
for metric in "${metrics[@]}"; do
  avg=$(grep "^${metric}" "$LOGFILE" | \
        awk '{print $NF}' | \
        awk '{ sum += $1; count++ } END { if (count > 0) print sum/count; else print "N/A" }')
  printf "%-30s %s\n" "$metric:" "$avg" | tee -a "$LOGFILE"
done
echo "=========================================" | tee -a "$LOGFILE"
