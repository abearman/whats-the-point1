#!/bin/bash
python solve.py --year=2012 --output=siftflow-real1-click1-cls-con --lr=1e-5 --train-gt=lmdb-real1-gt3 --val-gt=lmdb-real1-gt3 --expectation --constraint --classes --location --siftflow --display=1 --gpu=2
