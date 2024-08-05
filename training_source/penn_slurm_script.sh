#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Setup project on NLPGPU
python3 -m venv venv
source ./venv/bin/activate
export HF_HOME="./hiatus_huggingface_cache"
pip3 install -r requirements.txt
pip install datadreamer.dev==0.20.0

mkdir -p penn_slurm_logs

# Run training on all folds (only use for evaluation on test at the end)
export sample="crossGenre" # crossGenre, perGenre-HRS1.1
echo "START TIME: $(date)"
python3 -u trainer.py "data/train-test-dev split/{split}/Official Query-candidate format/" "cross_genre_all_v2" "train" "None" "dev" "None" "supervised_contrastive_loss" > penn_slurm_logs/train_cross_genre_all.stdout 3>&1 2>&1
echo "END TIME: $(date)"
