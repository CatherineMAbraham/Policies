#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=1      # 4 CPUs per agent
#SBATCH --mem=8G              # 8GB RAM per agent
#SBATCH --array=1-9
#SBATCH --time=10:00:00
#SBATCH --output=out_%A_%a.out


module load Anaconda3/2024.02-1

source activate softsurg
# Read the correct line from params_curr_compare.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" tests.csv)
IFS=',' read -r TISSUE YOUNGS_MODULUS NUM_SPRINGS <<< "$PARAM_LINE"
echo "Running test with: Tissue=$TISSUE, Young's Modulus=$YOUNGS_MODULUS, Number of Springs=$NUM_SPRINGS"
# Run the script 
srun --export=ALL python td3.py --threshold_pos 0.001 --threshold_ori 5 --action_type euler --maxforce 4 --num_springs $NUM_SPRINGS --softtissue $TISSUE --youngs_modulus $YOUNGS_MODULUS --contact_type 0 --ran $TASK_ID --log 1
