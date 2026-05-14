#!/bin/bash
#SBATCH --mail-user=cmabraham1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1            # 4 agents total
#SBATCH --cpus-per-task=10      # 4 CPUs per agent
#SBATCH --mem=20G              # 8GB RAM per agent
#SBATCH --time=02:00:00

module load Anaconda3/2024.02-1
source activate softsurg

# Run the script
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
PARAM_LINE=$(sed -n "${TASK_ID}p" model_log.csv)
IFS=',' read -r MODEL <<< "$PARAM_LINE"
MODEL=${MODEL//\'/}
# Convert relative path to absolute
if [[ "$MODEL" == /* ]]; then
    FULL_MODEL_PATH=/home/catherine/Policies2/TD3_Alg"$MODEL"
else
    FULL_MODEL_PATH=$MODEL
fi
echo "Testing model: $FULL_MODEL_PATH"
#srun --export=ALL 
python env_test2.py --num_eps 10000 --n_envs 1 --model_path "$FULL_MODEL_PATH" --maxforce 5 --softtissue soft --youngs_modulus 1e7 --log 1