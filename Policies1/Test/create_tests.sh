#!/bin/bash

# Create CSV file with header
> tests.csv

Generate combinations of tissue option, number of springs
for file in "rect00125.vtk"; do
    for ym in "1e7"; do
        for expert in "trajectory_1" "trajectory_2" "trajectory_3" "trajectory_4" "trajectory_5"; do
            echo "$file,$ym,$expert" >> tests.csv
        done
    done
done

echo "CSV file 'tests.csv' created successfully!"
cat tests.csv