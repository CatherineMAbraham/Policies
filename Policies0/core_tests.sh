#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=2      # 4 CPUs per agent
#SBATCH --mem=8G              # 8GB RAM per agent
#SBATCH --time=96:00:00
#SBATCH --array=1-8
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err

module load Anaconda3/2024.02-1

source activate softsurg

# Read the correct line from params_curr_compare.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" params2.csv)
#PARAM_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" params2.csv)
IFS=',' read -r THRESH_POS THRESH_ORI ACTION_TYPE<<< "$PARAM_LINE"

# Run the script
#srun --export=ALL 
python td3.py --threshold_pos $THRESH_POS --threshold_ori $THRESH_ORI --action_type $ACTION_TYPE --seed 1
