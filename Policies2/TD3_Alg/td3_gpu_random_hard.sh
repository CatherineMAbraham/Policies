#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=gpu
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=5     # 4 CPUs per agent
#SBATCH --array=1-10
#SBATCH --mem=30G              # 8GB RAM per agent
#SBATCH --time=25:00:00
#SBATCH --output=out_%A_%a.out


PYTHON_EXEC="/users/cop21cma/.conda/envs/softsurg9/bin/python"
#source activate softsurg
# Read the correct line from params_curr_compare.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" tests_params.csv)
IFS=',' read -r CONTACT  SEED <<< "$PARAM_LINE"
srun --export=ALL $PYTHON_EXEC td3_soft.py \
                --threshold_pos 0.0002 \
                --threshold_ori 0.2 \
                --action_type euler \
                --maxforce 5 \
                --num_springs 3 \
                --youngs_modulus 5e5 \
                --maximum_contact_force_threshold 0.25 \
                --softtissue spring \
                --render_mode 'direct' \
                --youngs_modulus_type testing \
                --randomise_num_springs 1 \
                --contact_type 1 \
                --seed $SEED \
                --ran $TASK_ID

