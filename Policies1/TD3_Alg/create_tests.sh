#!/bin/bash

# Create CSV file with header
> tests.csv

Generate combinations of tissue option, number of springs
for tissue in "spring" "None"; do
    if [ "$tissue" = "spring" ]; then
        options=("1" "3" "5" "10")
        ymoptions=("5e5" "1e6" "1.5e6" "5e6" "1e7")
    else
        options=("1")
        ymoptions=("1")
    fi

    for num in "${options[@]}"; do
        for ym in "${ymoptions[@]}"; do
            for seed in "5"; do
                echo "$tissue,$num,$ym,$seed" >> tests.csv
            done
        done
    done
done

echo "CSV file 'tests.csv' created successfully!"
cat tests.csv