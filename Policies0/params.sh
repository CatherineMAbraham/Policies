#!/bin/bash

> params.csv

# First three pairs with both fouractions and euler
pos_ori_pairs_both=("0.001 1" "0.0005 0.5" "0.0001 0.1" "0.00005 0.05" "0.00001 0.01")
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

