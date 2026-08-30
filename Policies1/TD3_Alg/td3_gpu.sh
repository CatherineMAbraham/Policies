#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=gpu
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=5      # 4 CPUs per agent
#SBATCH --mem=80G              # 8GB RAM per agent
#SBATCH --array=1-9
#SBATCH --time=25:00:00
#SBATCH --output=out_%A_%a.out


PYTHON_EXEC="/users/cop21cma/.conda/envs/softsurg9/bin/python"
#source activate softsurg9
# Read the correct line from params_curr_compare.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" tests.csv)
IFS=',' read -r TISSUE YOUNGS_MODULUS  SEED <<< "$PARAM_LINE"
# echo "Running test with: Tissue=$TISSUE, Young's Modulus=$YOUNGS_MODULUS, Seed=$SEED"
# Run the script 
srun --export=ALL $PYTHON_EXEC td3_v1.py \
  --threshold_pos 0.0005 \
  --threshold_ori 0.5 \
  --action_type euler \
  --maxforce 5 \
  --softtissue $TISSUE \
  --num_springs 1 \
  --youngs_modulus $YOUNGS_MODULUS \
  --contact_type 0 \
  --seed $SEED \
  --log 1 \
  --render_mode 'direct'
