#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=10            # 4 agents total
#SBATCH --cpus-per-task=1      # 4 CPUs per agent
#SBATCH --mem=80G              # 8GB RAM per agent
#SBATCH --time=40:00:00



module load Anaconda3/2024.02-1

source activate softsurg
wandb init --entity cmabraham1-university-of-sheffield --project Chp2-Sweep
# Run the script 
srun --export=ALL python td3_sweep.py 
