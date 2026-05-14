#!/bin/bash

# Create CSV file with header
> tests.csv

Generate combinations of tissue option, number of springs
for tissue in "spring"; do
    #if [ "$tissue" = "spring" ]; then
    options=("5" "10")
    if ["$options" = "5"]; then
        ymoptions=("1e7")
    else
        options=("10")
        ymoptions=("1e6" "5e6" "1e7")
    fi

    for num in "${options[@]}"; do
        for ym in "${ymoptions[@]}"; do
            for seed in "1" "2" "3" "4" "5" "6" "7" "8" "9" "10"; do
                echo "$tissue,$num,$ym,$seed" >> tests.csv
            done
        done
    done
done

echo "CSV file 'tests.csv' created successfully!"
cat tests.csv