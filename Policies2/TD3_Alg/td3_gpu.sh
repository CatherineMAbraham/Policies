#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=1      # 4 CPUs per agent
#SBATCH --mem=8G              # 8GB RAM per agent
#SBATCH --array=1-30
#SBATCH --time=60:00:00
#SBATCH --output=out_%A_%a.out


module load Anaconda3/2024.02-1

source activate softsurg
# Read the correct line from params_curr_compare.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" tests.csv)
IFS=',' read -r TISSUE NUM_SPRINGS YOUNGS_MODULUS  SEED <<< "$PARAM_LINE"
echo "Running test with: Tissue=$TISSUE, Young's Modulus=$YOUNGS_MODULUS, Number of Springs=$NUM_SPRINGS, Seed=$SEED"
# Run the script 
srun --export=ALL python td3.py --threshold_pos 0.001 --threshold_ori 5 --action_type euler --maxforce 4 --num_springs $NUM_SPRINGS --softtissue $TISSUE --youngs_modulus $YOUNGS_MODULUS --contact_type 0 --seed $SEED --ran $TASK_ID
