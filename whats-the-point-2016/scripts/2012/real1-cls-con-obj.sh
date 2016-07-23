#!/bin/bash
python solve.py --year=2012 --output=real1-cls-con-obj --lr=5e-5 --train-gt=lmdb-real1-gt3 --expectation --location --objectness --constraint --classes --gpu=2 --display=1
