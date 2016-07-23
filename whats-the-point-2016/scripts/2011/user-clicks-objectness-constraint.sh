#!/bin/bash
python solve.py --year=2011 --output=user-clicks-objectness-constraint --train-gt=lmdb-pascal_2011t-real1-gt-plus-objectness-3channel --expectation --location --objectness --constraint --gpu=2 --lr=1e-10 --display=1
