#!/bin/bash
python solve.py --year=2012 --output=bbox --lr=1e-5 --train-gt=lmdb-bbox-gt3 --expectation --location --constraint --display=1 --gpu=2 
