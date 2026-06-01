#!/bin/bash

> params.csv

# First three pairs with both fouractions and euler
pos_ori_pairs_both=("0.00015 1" "0.001 0.15")
for pair in "${pos_ori_pairs_both[@]}"; do
  pos=$(echo $pair | cut -d' ' -f1)
  ori=$(echo $pair | cut -d' ' -f2)
  for act in euler; do
      for seed in 1; do
          echo "$pos,$ori,$act,$seed" >> params.csv
      done
  done
done
     

# Last two pairs with only euler
# pos_ori_pairs_euler=("0.00008 1" "0.0001 0.8")
# for pair in "${pos_ori_pairs_euler[@]}"; do
#   pos=$(echo $pair | cut -d' ' -f1)
#   ori=$(echo $pair | cut -d' ' -f2)
#   for act in euler fouractions; do
#       for seed in 1; do
#           echo "$pos,$ori,$act,$seed" >> params.csv
#       done
#   done
# done

