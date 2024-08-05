#!/bin/bash
#
#SBATCH --time=168:00:00
#SBATCH --partition=p_nlp
#SBATCH --job-name=pausit-training
#SBATCH --output=penn_slurm_training.stdout
#SBATCH --constraint=48GBgpu
#SBATCH --gpus=1
#SBATCH --mem=30G

# Run the program
srun ./penn_slurm_script.sh
