#!/bin/bash
python solve.py --year=2012 --output=image-level-labels-constraint --lr=1e-5 --train-gt=lmdb-bbox-gt3 --expectation --constraint --gpu=2 --display=1 --momentum=0.9

#lmdb-bbox-gt3 
