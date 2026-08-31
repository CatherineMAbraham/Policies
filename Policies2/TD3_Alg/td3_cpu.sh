#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=5    # 4 CPUs per agent
#SBATCH --mem=40G              # 8GB RAM per agent
#SBATCH --array=1-10
#SBATCH --time=15:00:00
#SBATCH --output=out_%A_%a.out


PYTHON_EXEC="/users/cop21cma/.conda/envs/softsurg9/bin/python"
#source activate softsurg
# Read the correct line from params_curr_compare.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
#PARAM_LINE=$(sed -n "${TASK_ID}p" tests_params.csv)
#IFS=',' read -r CONTACT CONTACTFORCE SEED <<< "$PARAM_LINE"
#echo "Running test with: Contact=$CONTACT, Contact Force=$CONTACTFORCE, Seed=$SEED"
#Run the script 
srun --export=ALL $PYTHON_EXEC td3_soft.py \
                --threshold_pos 0.0005 \
                --threshold_ori 0.5 \
                --action_type euler \
                --maxforce 5 \
                --num_springs 3 \
                --youngs_modulus 5e5 \
                --maximum_contact_force_threshold 0.2 \
                --softtissue spring \
                --render_mode 'human' \
                --youngs_modulus_type None \
                --log 1 \
                --contact_type 0 \
                --seed $TASK_ID \
                --ran 1 \
