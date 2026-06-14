#!/bin/bash

# Create CSV file with header
> tests.csv

Generate combinations of tissue option, number of springs
for file in "rect0009.vtk" "rect0008.vtk" "rect0007.vtk" "rect0006.vtk"; do
    for ym in "1e5"; do
        for expert in "trajectory_2"; do
            echo "$file,$ym,$expert" >> tests.csv
        done
    done
done

echo "CSV file 'tests.csv' created successfully!"
cat tests.csv