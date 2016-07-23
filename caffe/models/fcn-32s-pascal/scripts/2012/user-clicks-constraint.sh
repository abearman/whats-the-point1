#!/bin/bash
python solve.py --year=2012 --output=user-clicks-constraint --lr=1e-5 --train-gt=lmdb-real1-gt3 --expectation --location --constraint --gpu=2 --display=1
