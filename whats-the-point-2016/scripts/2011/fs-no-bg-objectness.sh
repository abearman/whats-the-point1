#!/bin/bash
python solve.py --year=2011 --output=fs-no-bg-objectness --lr=1e-10 --train-gt=lmdb-fs-minus-bg-gt3 --expectation --location --objectness --gpu=3 
