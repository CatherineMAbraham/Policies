#!/bin/bash

> params.csv

# First three pairs with both fouractions and euler
pos_ori_pairs_both=("0.00009 0.9" "0.00008 0.8" "0.00007 0.7" "0.00006 0.6" "0.00005 0.5" "0.00004 0.4" "0.00003 0.3" "0.00002 0.2" "0.00001 0.1" "0.000009 0.09" "0.000008 0.08" "0.000007 0.07" "0.000006 0.06" "0.000005 0.05" "0.000004 0.04" "0.000003 0.03" "0.000002 0.02" "0.000001 0.01")
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
# pos_ori_pairs_euler=("0.0005 1" "0.002 0.25")
# for pair in "${pos_ori_pairs_euler[@]}"; do
#   pos=$(echo $pair | cut -d' ' -f1)
#   ori=$(echo $pair | cut -d' ' -f2)
#   echo "$pos,$ori,euler" >> params_pos.csv
# done

