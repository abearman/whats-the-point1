#!/bin/bash
python solve.py --year=2012 --output=image-level-labels-just-objectness --lr=5e-6 --train-gt=lmdb-bbox-gt3 --expectation --location --objectness --gpu=2 --display=1 --momentum=0.9

#lmdb-bbox-gt3 
