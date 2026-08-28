#!/bin/bash

# Create CSV file with header
 > tests_params.csv

# for ym_type in "None" "testing"; do
#     if [ "$ym_type" = "None" ]; then
#         contact=1
#     else
#         contact=0
#     fi

#     for seed in {1..5}; do
#         echo "$ym_type,$contact,$seed" >> tests_params.csv
#     done
# done
#for contact in 1 0; do
for contact_threshold in 0.1 0.2 0.25 0.5; do
    for contact in 1 0; do
        for seed in {1..3}; do
            echo "$contact,$contact_threshold,$seed" >> tests_params.csv
        done
    done
done
    

echo "CSV file 'tests_params.csv' created successfully!"
cat tests_params.csv