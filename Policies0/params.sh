#!/bin/bash

> params2.csv

# First three pairs with both fouractions and euler
pos_ori_pairs_both=("0.0005 0.25" "0.001 0.5" "0.002 1")
for pair in "${pos_ori_pairs_both[@]}"; do
  pos=$(echo $pair | cut -d' ' -f1)
  ori=$(echo $pair | cut -d' ' -f2)
  for act in fouractions euler; do
      echo "$pos,$ori,$act" >> params2.csv
  done
done

# Last two pairs with only euler
pos_ori_pairs_euler=("0.0005 1" "0.002 0.25")
for pair in "${pos_ori_pairs_euler[@]}"; do
  pos=$(echo $pair | cut -d' ' -f1)
  ori=$(echo $pair | cut -d' ' -f2)
  echo "$pos,$ori,euler" >> params2.csv
done

