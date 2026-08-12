#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=4      # 4 CPUs per agent
#SBATCH --mem=8G              # 8GB RAM per agent
#SBATCH --array=1-20
#SBATCH --time=96:00:00
#SBATCH --output=out_%A_%a.out

module load Anaconda3/2024.02-1

source activate softsurg
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" tests_params.csv)
IFS=',' read -r YM_TYPE CONTACT SEED <<< "$PARAM_LINE"
echo "Running test with: Young's Modulus Type=$YM_TYPE, Contact Type=$CONTACT, Seed=$SEED"
# Run the script
srun --export=ALL python td3_soft.py --threshold_pos 0.0002 --threshold_ori 0.2 --action_type euler --maxforce 3 --softtissue spring --youngs_modulus_type $YM_TYPE --contact_type $CONTACT --seed $SEED --render_mode 'direct' --log 1 --ran $TASK_ID
