#!/bin/bash

#SBATCH --account=dsi            # The account name for the job.
#SBATCH --job-name=pausit-training  # The job name.
#SBATCH --output=penn_slurm_training.stdout
#SBATCH --gres=gpu:1             # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH --time=0-8:00           # The time the job will take to run in D-HH:MM
#SBATCH --mem=64G


# Run the program
#srun ./penn_slurm_script.sh
#module load cuda11.8/toolkit
#python3 -m venv venv
#source ./venv/bin/activate
#export HF_HOME="./hiatus_huggingface_cache"
#pip3 install -r requirements.txt
#pip install datadreamer.dev==0.36
jupyter notebook --no-browser --ip=$(hostname -i) --notebook-dir=/burg/old_dsi/users/ma4608
