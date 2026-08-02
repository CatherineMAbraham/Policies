#!/bin/bash

# Create CSV file with header
echo "ym_type,num,seed" > tests_params.csv

for ym_type in "None" "testing"; do
    if [ "$ym_type" = "None" ]; then
        contact=1
    else
        contact=0
    fi

    for seed in {1..10}; do
        echo "$ym_type,$contact,$seed" >> tests_params.csv
    done
done

echo "CSV file 'tests_params.csv' created successfully!"
cat tests_params.csv