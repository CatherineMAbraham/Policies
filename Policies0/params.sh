#!/bin/bash

> params_pos.csv

# First three pairs with both fouractions and euler
pos_ori_pairs_both=("0.0005 0.0" "0.00025 0.0" "0.000125 0" "0.00006 0" "0.00001 0" "0 0")
for pair in "${pos_ori_pairs_both[@]}"; do
  pos=$(echo $pair | cut -d' ' -f1)
  ori=$(echo $pair | cut -d' ' -f2)
  for act in pos_only; do
      echo "$pos,$ori,$act" >> params_pos.csv
  done
done

# Last two pairs with only euler
# pos_ori_pairs_euler=("0.0005 1" "0.002 0.25")
# for pair in "${pos_ori_pairs_euler[@]}"; do
#   pos=$(echo $pair | cut -d' ' -f1)
#   ori=$(echo $pair | cut -d' ' -f2)
#   echo "$pos,$ori,euler" >> params_pos.csv
# done

