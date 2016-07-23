#!/bin/bash
python solve.py --year=2011 --output=fs-no-bg --lr=1e-10 --train-gt=lmdb-fs-minus-bg-gt3 --expectation --location --gpu=2 
