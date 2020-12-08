#!/bin/bash

if [ ! -d outputs ]; then
    mkdir outputs
fi

./RunBench.py fraud_detection_1 > outputs/fraud_detection_1.txt

./RunBench.py insurance_1 --num_tests 1000 > outputs/insurance_1.txt

./RunBench.py categorical_encoding_1 --num_tests 1000 > outputs/categorical_encoding_1.txt
