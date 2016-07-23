#!/bin/bash
python solve.py --year=2012 --output=real1-click1-con-obj --lr=1e-5 --train-gt=lmdb-real1_click1-gt3 --expectation --location --objectness --constraint --gpu=1 --display=1
