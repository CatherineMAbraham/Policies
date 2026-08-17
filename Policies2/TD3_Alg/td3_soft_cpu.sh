#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=4      # 4 CPUs per agent
#SBATCH --mem=80G              # 8GB RAM per agent
#SBATCH --array=1-2
#SBATCH --time=40:00:00
#SBATCH --output=out_%A_%a.out

PYTHON_EXEC="/users/cop21cma/.conda/envs/softsurg9/bin/python"

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" tests_params.csv)
IFS=',' read -r YM_TYPE FORCE SEED <<< "$PARAM_LINE"
echo "Running test with: Young's Modulus Type=$YM_TYPE, Contact Type=$CONTACT, Seed=$SEED"
# Run the script
srun --export=ALL $PYTHON_EXEC td3_soft.py --threshold_pos 0.0005 --threshold_ori 0.5 --action_type euler --maxforce $FORCE --softtissue spring --youngs_modulus_type $YM_TYPE --contact_type 0 --seed $SEED --render_mode 'direct' --log 1 --ran $TASK_ID
