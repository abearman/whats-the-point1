#!/bin/bash
python solve.py --year=2012 --output=image-level-labels --lr=1e-5 --train-gt=lmdb-bbox-gt3 --expectation --gpu=1 --display=1 --momentum=0.9

#lmdb-bbox-gt3 
