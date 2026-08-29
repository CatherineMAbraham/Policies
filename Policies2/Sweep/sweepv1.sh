#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1        # 4 agents total
#SBATCH --cpus-per-task=1     # 4 CPUs per agent
#SBATCH --mem=20G              # 8GB RAM per agent
#SBATCH --array=1-20
#SBATCH --time=48:00:00
#SBATCH --output=out_%A_%a.out


PYTHON_EXEC="/users/cop21cma/.conda/envs/softsurg9/bin/python"






wandb init --entity cmabraham1-university-of-sheffield --project Chapter3-Sweep
# Run the script 
SWEEP_ID="fe84pfv9"
srun --export=ALL $PYTHON_EXEC td3_sweep_v2.py --sweep_id $SWEEP_ID --count 5
