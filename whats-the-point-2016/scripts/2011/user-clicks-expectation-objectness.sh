#!/bin/bash
python solve.py --year=2011 --output=user-clicks-expectation-objectness --train-gt=lmdb-pascal_2011t-real1-gt-plus-objectness-3channel --expectation --location --objectness --gpu=3 --lr=1e-10

