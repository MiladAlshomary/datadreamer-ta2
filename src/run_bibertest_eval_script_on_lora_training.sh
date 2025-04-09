#!/bin/bash

# Expect user to pass something like:
# /data/araghavan/HIATUS/datadreamer-ta2/src/run_eval_script_on_lora_training.sh sadiri_hardpos_rankingloss_random_hardbatching_v001/20250402_193508
# Check if argument was passed
if [ -z "$1" ]; then
  echo "Usage: $0 <BASE_DIR_NAME>"
  echo "Example: $0 sadiri_hardpos_rankingloss_random_hardbatching_v001_20250402_193508 gte70/gte250"
  exit 1
fi

BASE_DIR_NAME="${1}"
BIBER_DOC_TYPE="${2}"

# Check if biber doc type is either gte70 or gte250
if [ "${BIBER_DOC_TYPE}" == "gte70" ]; then
  echo "Using BIBER doc type: gte70"
elif [ "${BIBER_DOC_TYPE}" == "gte250" ]; then
  echo "Using BIBER doc type: gte250"
else
  echo "Invalid BIBER doc type. Use either 'gte70' or 'gte250'."
  exit 1
fi

# Remove slashes → underscores
BASE_DIR_NAME_NOSLASH=$(echo "$BASE_DIR_NAME" | tr '/' '_')

# Paths
MODEL_CHECKPOINT="/data/araghavan/HIATUS/datadreamer-ta2/output/${BASE_DIR_NAME}/luar-trainer/_model"
PAUSIT_EVAL_TA2_DIR="/data/araghavan/HIATUS/pausit-eval-ta2"
LORA_CHECKPOINT="${PAUSIT_EVAL_TA2_DIR}/SIVs/datadreamer_lora/checkpoint"
BASE_INPUT_DIR="/data/araghavan/HIATUS/datadreamer-ta2/data/biber/qc_samples_${BIBER_DOC_TYPE}/TA2/biber_test_qc"

# Derived paths
BASE_OUTPUT_DIR="/data/araghavan/HIATUS/datadreamer-ta2/data/biber/qc_samples_${BIBER_DOC_TYPE}_outputs/bibertest_${BASE_DIR_NAME}"
LOGFILE="/data/araghavan/HIATUS/datadreamer-ta2/output/logs/bibertest_${BIBER_DOC_TYPE}_${BASE_DIR_NAME_NOSLASH}_eval_op.txt"

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

echo "==== Evaluation Started at $(date) ===="
echo "Writing to log file: ######## $LOGFILE ########"

echo "==== Evaluation Started at $(date) ====" > "$LOGFILE"
echo "Using BASE_OUTPUT_DIR: $BASE_OUTPUT_DIR" >> "$LOGFILE"
cd "${PAUSIT_EVAL_TA2_DIR}" || exit 1

RUN_ID=14
INPUT_DIR="${BASE_INPUT_DIR}/data/"
GROUNDTRUTH_DIR="${BASE_INPUT_DIR}/groundtruth/"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/"

echo "==== Running Biber QC Test ${BIBER_DOC_TYPE} Eval (RUN_ID=${RUN_ID}) ===="
echo "==== Running Biber QC Test ${BIBER_DOC_TYPE} Eval (RUN_ID=${RUN_ID}) ====" >> "$LOGFILE"
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
  echo -e "\t\t==== Eval Completed ====\n" | tee -a "$LOGFILE"
else
  echo -e "\t\t❌ Eval FAILED!\n" | tee -a "$LOGFILE"
fi

echo "==== Eval Finished at $(date) ====" >> "$LOGFILE"
echo "Final Output Directory: $BASE_OUTPUT_DIR" >> "$LOGFILE"

echo "==== Displaying metrics... ====" | tee -a "$LOGFILE"

# Print all metrics directly from the log file
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

for metric in "${metrics[@]}"; do
#   echo "==== $metric ====" | tee -a "$LOGFILE"
  grep "^$metric" "$LOGFILE" | tee -a "$LOGFILE"
done

echo "Log File: $LOGFILE"
