#!/bin/bash
python solve.py --year=2012 --output=bbox-unnormalized --lr=1e-10 --train-gt=lmdb-bbox-gt3 --expectation --location --objectness --constraint --no_norm_sup --display=1 --gpu=2 
