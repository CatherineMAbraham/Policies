#!/bin/bash
source activate softsurg

# Read the correct line from params_all.csv
##get num lines in csv
NUM_LINES=$(wc -l < params_sac.csv)
for i in $(seq 1 $NUM_LINES); do
    PARAM_LINE=$(sed -n "${i}p" params_sac.csv)
    IFS=',' read -r MODEL REWARD SEED <<< "$PARAM_LINE"
    echo "Running with MODEL=$MODEL, REWARD=$REWARD, SEED=$SEED"
    python alg_compare.py --threshold_pos 0.0002 --threshold_ori 0.2 --action_type 'euler' --seed $SEED --reward $REWARD --model $MODEL --ran $i
done
