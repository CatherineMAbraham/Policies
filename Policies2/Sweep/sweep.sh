#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=gpu
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=out_%A_%a.out

# Activate Conda Environment properly
source ~/.bashrc
conda activate softsurg9

# (Optional) Export W&B API key if needed
# export WANDB_API_KEY="your_api_key_here"

# Execute agent script with the generated Sweep ID
SWEEP_ID="v8avoer6"

python td3_sweep_v2.py --sweep_id $SWEEP_ID --count 5