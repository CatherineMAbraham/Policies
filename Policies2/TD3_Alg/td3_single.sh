#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=5      # 4 CPUs per agent
#SBATCH --mem=80G              # 8GB RAM per agent
#SBATCH --time=30:00:00
#SBATCH --output=out_%A_%a.out


module load Anaconda3/2024.02-1

source activate softsurg9
# Read the correct line from params_curr_compare.csv

# Run the script
#srun --export=ALL 
python td3_cl.py \
    --threshold_pos 0.0005 \
    --threshold_ori 0.5 \
    --action_type euler \
    --maxforce 5 \
    --maximum_contact_force_threshold 0.4 \
    --num_springs 3 \
    --youngs_modulus 5e5 \
    --softtissue spring \
    --render_mode 'direct' \
    --youngs_modulus_type None \
    --contact_type 1 \
    --log 1 \
    --seed 1 \
    --ran 1
