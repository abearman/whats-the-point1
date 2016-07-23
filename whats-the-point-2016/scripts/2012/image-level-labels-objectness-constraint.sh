#!/bin/bash
python solve.py --year=2012 --output=image-level-labels-objectness-constraint --lr=1e-5 --train-gt=lmdb-bbox-gt3 --expectation --objectness --constraint --momentum=0.9 --gpu=3 --display=1

#lmdb-bbox-gt3 
