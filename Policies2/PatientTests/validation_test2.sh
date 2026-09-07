#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=1    # 4 CPUs per agent
#SBATCH --mem=20G              # 8GB RAM per agent
#SBATCH --array=1-20
#SBATCH --time=5:00:00
#SBATCH --output=out_%A_%a.out


# Read the correct line from params_curr_compare.csv
PYTHON_EXEC="/users/cop21cma/.conda/envs/softsurg9/bin/python"
#source activate softsurg
# Read the correct line from params_curr_compare.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
#PARAM_LINE=$(sed -n "${TASK_ID}p" tests_params.csv)
PARAM_LINE=$(sed -n "${TASK_ID}p" tests.csv)
IFS=',' read -r MODEL SAFE SEED<<< "$PARAM_LINE"

srun --export=ALL $PYTHON_EXEC validation_tests_random.py \
                --model_path $MODEL \
                --log 1\
                --maxforce 5\
                --safemode 0\
                --num_eps 1000\
                --force_limit 0.4\
                --randomise_start 1\
                --seed $SEED \