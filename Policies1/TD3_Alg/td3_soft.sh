#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=4      # 4 CPUs per agent
#SBATCH --mem=8G              # 8GB RAM per agent
#SBATCH --time=96:00:00
#SBATCH --output=out_%A_%a.out

module load Anaconda3/2024.02-1

source activate softsurg
# Run the script
#srun --export=ALL 
python td3_soft.py --threshold_pos 0.001 --threshold_ori 5 --action_type euler --maxforce 3.5 --softtissue soft
