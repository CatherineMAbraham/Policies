#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=1      # 4 CPUs per agent
#SBATCH --mem=20G              # 8GB RAM per agent
#SBATCH --time=40:00:00
#SBATCH --output=out_%A_%a.out


module load Anaconda3/2024.02-1

source activate softsurg
# Read the correct line from params_curr_compare.csv

# Run the script
#srun --export=ALL 
python td3_contact.py --threshold_pos 0.0002 --threshold_ori 0.2 --action_type euler --maxforce 3 --youngs_modulus_type None --youngs_modulus 1e6  --softtissue spring --contact_type 1 --render_mode 'direct'  --log 1
