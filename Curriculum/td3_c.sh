#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=1      # 4 CPUs per agent
#SBATCH --mem=8G              # 8GB RAM per agent
#SBATCH --time=10:00:00
#SBATCH --output=out_%A_%a.out


module load Anaconda3/2024.02-1

source activate softsurg
# Read the correct line from model_paths.csv
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
MODEL_LINE=$(sed -n "${TASK_ID}p" model_paths.csv)
IFS=',' read -r MODEL_PATH SEED <<< "$MODEL_LINE"
echo "Running test with model path: $MODEL_PATH"
MODEL=${MODEL_PATH//\'/}
#Convert relative path to absolute
if [[ "$MODEL" == /* ]]; then
    FULL_MODEL_PATH=/users/cop21cma/Policies2/TD3_Alg"$MODEL"
else
    FULL_MODEL_PATH=$MODEL
fi
echo "Full model path: $FULL_MODEL_PATH"
echo "Seed: $SEED"
# Run the script
srun --export=ALL python td3_curriculum.py --threshold_pos 0.001 --threshold_ori 2 --action_type euler --maxforce 4 --softtissue spring --num_springs 3  --youngs_modulus 1e7 --model "$FULL_MODEL_PATH" --contact_type 0 --log 1 --seed $SEED