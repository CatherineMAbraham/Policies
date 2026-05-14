#!/bin/bash

# Create CSV file with header
> tests.csv

# Generate combinations of tissue option, number of springs
for tissue in "spring"; do
    if [ "$tissue" = "spring" ]; then
        options=("1" "3")
    else
        options=("1")
    fi

    for num in "${options[@]}"; do
        for ym in "1e7" "5e7" "2e7" "1e8"; do
            echo "$tissue,$num,$ym" >> tests.csv
        done
    done
done

echo "CSV file 'tests.csv' created successfully!"
cat tests.csv