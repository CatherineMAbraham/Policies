#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=1     # 4 CPUs per agent
#SBATCH --mem=8G              # 8GB RAM per agent
#SBATCH --time=20:00:00
#SBATCH --array=1-54
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err

module load Anaconda3/2024.02-1

source activate softsurg

# Read the correct line from params_all.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" params_all.csv)
IFS=',' read -r MODEL REWARD SEED <<< "$PARAM_LINE"

# Run the script
srun --export=ALL python alg_compare.py --threshold_pos 0.0005 --threshold_ori 0.5 --action_type 'euler' --seed $SEED --reward $REWARD --model $MODEL --ran $TASK_ID
