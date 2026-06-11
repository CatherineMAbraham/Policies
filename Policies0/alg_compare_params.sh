#!/bin/bash

# Create CSV file with header
> params_all.csv

# Generate combinations of models, reward options, and numbers 1-3
for model in "SAC" "TD3"; do
    for option in "sparse"; do
        for num in {1..10}; do
            echo "$model,$option,$num" >> params_all.csv
        done
    done
done


echo "CSV file 'params_all.csv' created successfully!"
cat params_all.csv
