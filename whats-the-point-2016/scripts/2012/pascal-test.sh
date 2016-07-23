#!/bin/bash
python solve.py --year=2012 --output=pascal-test --lr=1e-5 --test --train-gt=lmdb-real1_click1-gt3 --val-gt=lmdb-real1_click1-gt3 --expectation --location --objectness --constraint --classes --display=1 --gpu=2
