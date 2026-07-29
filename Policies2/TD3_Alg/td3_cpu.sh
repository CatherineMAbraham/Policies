#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=5      # 4 CPUs per agent
#SBATCH --mem=20G              # 8GB RAM per agent
#SBATCH --array=1-8
#SBATCH --time=20:00:00
#SBATCH --output=out_%A_%a.out


module load Anaconda3/2024.02-1

source activate softsurg
# Read the correct line from params_curr_compare.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" tests_params.csv)
IFS=',' read -r SCRIPT YOUNGSTYPE RANDOMISE_LIGS RANDOMISE_START<<< "$PARAM_LINE"

# Run the script
srun --export=ALL python $SCRIPT --threshold_pos 0.0005 --threshold_ori 0.5 --action_type euler --maxforce 5 --youngs_modulus_type $YOUNGSTYPE --randomise_ligs $RANDOMISE_LIGS --randomise_start $RANDOMISE_START --youngs_modulus 1e6 --softtissue spring --contact_type 0 --render_mode 'direct'  --log 1