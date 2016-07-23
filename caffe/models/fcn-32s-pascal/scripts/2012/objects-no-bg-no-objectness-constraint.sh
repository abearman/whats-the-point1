#!/bin/bash
python solve.py --year=2012 --output=objects-no-bg-no-objectness-constraint.sh --lr=1e-10 --train-gt=lmdb-pascal_2011t-gt-ignorebg-plus-objectness --expectation --location --constraint --gpu=0 
