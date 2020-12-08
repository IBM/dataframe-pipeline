#!/bin/bash

dirs=("fraud_detection_1" "insurance_1" "categorical_encoding_1" "mental_health_1")
competitions=("ieee-fraud-detection" "homesite-quote-conversion" "cat-in-the-dat" "osmi/mental-health-in-tech-survey")

for i in {0..3}
do
    dir=${dirs[$i]}
    competition=${competitions[$i]}
    
    echo "Downloading $competition to inputs/$dir"

    if [ $dir = "mental_health_1" ]; then
        kaggle datasets download -p inputs/$dir $competition
    else
        kaggle competitions download -p inputs/$dir $competition
    fi
    mkdir -p inputs/$dir
    cd inputs/$dir
    unzip $competition.zip
    rm -rf $competition.zip
    files="*.zip"
    for file in $files
    do
        unzip $file
        rm -rf $file
    done
    cd ../../

done


