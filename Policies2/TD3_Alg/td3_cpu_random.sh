#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=5     # 4 CPUs per agent
#SBATCH --array=1-20
#SBATCH --mem=30G              # 8GB RAM per agent
#SBATCH --time=25:00:00
#SBATCH --output=out_%A_%a.out


#PYTHON_EXEC="/users/cop21cma/.conda/envs/softsurg9/bin/python"
source activate softsurg
# Read the correct line from params_curr_compare.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
#PARAM_LINE=$(sed -n "${TASK_ID}p" tests_params.csv)
#IFS=',' read -r CONTACT_THRESHOLD SEED <<< "$PARAM_LINE"
#srun --export=ALL $PYTHON_EXEC 
python td3_soft.py \
                --threshold_pos 0.0005 \
                --threshold_ori 0.5 \
                --action_type euler \
                --maxforce 5 \
                --num_springs 3 \
                --youngs_modulus 5e5 \
                --maximum_contact_force_threshold 1.0 \
                --softtissue spring \
                --render_mode 'direct' \
                --youngs_modulus_type testing \
                --randomise_foot_dynamics 1\
                --randomise_num_springs 0 \
                --contact_type 1 \
                --seed 1 \
                --ran 1

