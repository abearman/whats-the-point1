#!/bin/bash
python solve.py --year=2012 --output=siftflow-labels-cls-con --lr=1e-5 --train-gt=lmdb-real1-gt3 --expectation --constraint --classes --siftflow --display=1 --gpu=3
