#!/bin/bash

# Create CSV file with header
> params_sac.csv
> params_td3.csv
# Generate combinations of models, reward options, and numbers 1-3
for model in "SAC"; do
    for option in "sparse"; do
        for num in {4..7}; do
            echo "$model,$option,$num" >> params_sac.csv
        done
    done
done

for model in "TD3"; do
    for option in "sparse"; do
        for num in {4..7}; do
            echo "$model,$option,$num" >> params_td3.csv
        done
    done
done
echo "CSV file 'params_all.csv' created successfully!"
cat params_all.csv
