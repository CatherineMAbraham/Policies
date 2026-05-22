#!/bin/bash

> params.csv

# First three pairs with both fouractions and euler
pos_ori_pairs_both=("0.000025 0.25" "0.00005 0.5" "0.0001 1")
for pair in "${pos_ori_pairs_both[@]}"; do
  pos=$(echo $pair | cut -d' ' -f1)
  ori=$(echo $pair | cut -d' ' -f2)
  for act in euler fouractions; do
      for seed in {1..3}; do
          echo "$pos,$ori,$act,$seed" >> params.csv
      done
  done
done
     

# Last two pairs with only euler
# pos_ori_pairs_euler=("0.0005 1" "0.002 0.25")
# for pair in "${pos_ori_pairs_euler[@]}"; do
#   pos=$(echo $pair | cut -d' ' -f1)
#   ori=$(echo $pair | cut -d' ' -f2)
#   echo "$pos,$ori,euler" >> params_pos.csv
# done

