#!/bin/bash

> params.csv

# First three pairs with both fouractions and euler
# pos=("0.001" "0.0005" "0.0001")
# for pair in "${pos[@]}"; do
#   pos=$(echo $pair | cut -d' ' -f1)
#   for act in pos_only; do
#       for seed in {1..10}; do
#           echo "$pos,$act,$seed" >> params_pos.csv
#       done
#   done
# done

pos_ori_pairs_both=("0.0002 0.2")
for pair in "${pos_ori_pairs_both[@]}"; do
  pos=$(echo $pair | cut -d' ' -f1)
  ori=$(echo $pair | cut -d' ' -f2)
  for act in euler; do
      for seed in {1..10}; do
          echo "$pos,$ori,$act,$seed" >> params.csv
      done
  done
done

# # Last two pairs with only euler
# pos=("0.001" "0.0005" "0.0001")
# for pair in "${pos[@]}"; do
#   pos=$(echo $pair | cut -d' ' -f1)
#   for act in pos_only; do
#       for seed in {1..10}; do
#           echo "$pos,$act,$seed" >> params.csv
#       done
#   done
# done

