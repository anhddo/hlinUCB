#!/bin/bash
python main.py -M 60 -d 30 -T 10000 -ds 50 -runs 2 -o dump -jobs 30 -eps 0.00
python main.py -M 60 -d 30 -T 10000 -ds 50 -runs 2 -o dump -jobs 30 -eps 0.00067
python main.py -M 60 -d 30 -T 10000 -ds 50 -runs 2 -o dump -jobs 30 -eps 0.00268
python main.py -M 60 -d 30 -T 10000 -ds 50 -runs 2 -o dump -jobs 30 -eps 0.27 -base-scale 0.75
python main.py -M 60 -d 30 -T 10000 -ds 50 -runs 2 -o dump -jobs 30 -eps 0.68 -base-scale 0.75
python main.py -M 60 -d 30 -T 10000 -ds 50 -runs 2 -o dump -jobs 30 -eps 0.76

python plot.py -path dump -o plot.pdf
