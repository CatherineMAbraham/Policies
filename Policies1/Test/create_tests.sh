#!/bin/bash

# Create CSV file with header
> tests.csv

Generate combinations of tissue option, number of springs
for file in "rect0001.vtk" "rect000125.vtk" "rect00025.vtk" "rect0005.vtk" "rect001.vtk" "rect00125.vtk" "rect0025.vtk" "rect005.vtk"; do
    for ym in "1e5"; do
        for expert in "trajectory_2"; do
            echo "$file,$ym,$expert" >> tests.csv
        done
    done
done

echo "CSV file 'tests.csv' created successfully!"
cat tests.csv