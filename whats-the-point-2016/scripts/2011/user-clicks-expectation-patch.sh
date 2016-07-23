#!/bin/bash
python solve.py --output=user-clicks-expectation-patch --train-gt=lmdb-pascal_2011t-real1-radius2-gt-plus-objectness-3channel --expectation=true --use-gpu=yes --gpu=1 --lr=1e-12

