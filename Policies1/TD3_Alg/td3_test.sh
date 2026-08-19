#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=5      # 4 CPUs per agent
#SBATCH --mem=80G              # 8GB RAM per agent
#SBATCH --array=1-10
#SBATCH --time=25:00:00
#SBATCH --output=out_%A_%a.out


module load Anaconda3/2024.02-1

source activate softsurg
#PYTHON_EXEC="/users/cop21cma/.conda/envs/softsurg9/bin/python"
# Read the correct line from params_curr_compare.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" model_log.csv)
IFS=',' read -r MODEL SEED<<< "$PARAM_LINE"
echo "Running test with: Model=$MODEL"
# Run the script 
srun --export=ALL python td3_v1.py \
 --threshold_pos 0.0005 \
  --threshold_ori 0.5 \
  --maxforce 4 \
 --model $MODEL \
 --seed $SEED \
  --log 1 \
  --render_mode 'direct'