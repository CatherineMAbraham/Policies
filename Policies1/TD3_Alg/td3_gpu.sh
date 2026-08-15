#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=5      # 4 CPUs per agent
#SBATCH --mem=10G              # 8GB RAM per agent
#SBATCH --array=1-10
#SBATCH --time=00:30:00
#SBATCH --output=out_%A_%a.out


source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.conda/etc/profile.d/conda.sh 2>/dev/null || source /usr/local/module/anaconda3/etc/profile.d/conda.sh

conda activate softsurg9
# Read the correct line from params_curr_compare.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" tests.csv)
IFS=',' read -r TISSUE YOUNGS_MODULUS  SEED <<< "$PARAM_LINE"
echo "Running test with: Tissue=$TISSUE, Young's Modulus=$YOUNGS_MODULUS, Seed=$SEED"
# Run the script 
srun --export=ALL python td3_v1_soft.py --threshold_pos 0.0005 --threshold_ori 0.5 --action_type euler --maxforce 3  --softtissue $TISSUE --youngs_modulus $YOUNGS_MODULUS --contact_type 0 --seed $SEED --ran $TASK_ID --log 1
