#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=1      # 4 CPUs per agent
#SBATCH --mem=8G              # 8GB RAM per agent
#SBATCH --array=1-4
#SBATCH --time=1:00:00
#SBATCH --output=out_%A_%a.out


module load Anaconda3/2024.02-1

source activate softsurg
# Read the correct line from params_curr_compare.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-16}
PARAM_LINE=$(sed -n "${TASK_ID}p" tests.csv)
IFS=',' read -r FILE YOUNGS_MODULUS EXPERT <<< "$PARAM_LINE"
echo "Running test with: Young's Modulus=$YOUNGS_MODULUS, VTK File=$FILE, Expert Trajectory=$EXPERT"
# Run the script 
#srun --export=ALL 
python force_from_expert.py --threshold_pos 0.0001 --threshold_ori 0.5 --maxforce 4 --softtissue soft --youngs_modulus $YOUNGS_MODULUS --vtk_file $FILE --expert $EXPERT --log 1
