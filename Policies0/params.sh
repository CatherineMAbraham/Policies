#!/bin/bash

> params.csv

# First three pairs with both fouractions and euler
pos_ori_pairs_both=("0.00005 0.05" "0.0001 0.1" "0.0002 0.2")
for pair in "${pos_ori_pairs_both[@]}"; do
  pos=$(echo $pair | cut -d' ' -f1)
  ori=$(echo $pair | cut -d' ' -f2)
  for act in euler fouractions; do
      echo "$pos,$ori,$act" >> params.csv
  done
done

# Last two pairs with only euler
# pos_ori_pairs_euler=("0.0005 1" "0.002 0.25")
# for pair in "${pos_ori_pairs_euler[@]}"; do
#   pos=$(echo $pair | cut -d' ' -f1)
#   ori=$(echo $pair | cut -d' ' -f2)
#   echo "$pos,$ori,euler" >> params_pos.csv
# done

