#!/bin/bash

# Create CSV file with header
> tests.csv

#Generate combinations of tissue option, number of springs
for file in "rect0005.vtk" "rect0006.vtk" "rect0007.vtk" "rect0008.vtk" "rect0009.vtk" "rect001.vtk" "rect00125.vtk" "rect0025.vtk" "rect005.vtk" "rect01.vtk" "None"; do
    for ym in "1e6"; do
        for expert in "Expert_2_actions"; do
            echo "$file,$ym,$expert" >> tests.csv
        done
    done
done

echo "CSV file 'tests.csv' created successfully!"
cat tests.csv