#!/bin/bash
python solve.py --year=2012 --output=objects-no-bg-objectness-constraint --lr=1e-10 --train-gt=lmdb-fs-minus-bg-gt3 --expectation --location --objectness --constraint --gpu=0 
