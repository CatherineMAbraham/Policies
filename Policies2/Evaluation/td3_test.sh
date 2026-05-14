#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=5     # 4 CPUs per agent
#SBATCH --array=1-9
#SBATCH --mem=20G              # 8GB RAM per agent
#SBATCH --time=07:00:00


module load Anaconda3/2024.02-1
source activate softsurg

# Run the script
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" model_log.csv)
IFS=',' read -r MODEL SEED<<< "$PARAM_LINE"
MODEL=${MODEL//\'/}
SEED=${SEED//\'/}
#Convert relative path to absolute
if [[ "$MODEL" == /* ]]; then
    FULL_MODEL_PATH=/users/cop21cma/Policies2/TD3_Alg"$MODEL"
else
    FULL_MODEL_PATH="$MODEL"
fi
echo "Testing model: $MODEL"
#srun --export=ALL 
python env_test2.py --num_eps 1000 --n_envs 1 --model_path "$FULL_MODEL_PATH" --maxforce 4 --softtissue spring --vtk_file rect005.vtk --threshold_pos 0.001 --threshold_ori 1 --youngs_modulus 1e7 --log 0 --seed $SEED