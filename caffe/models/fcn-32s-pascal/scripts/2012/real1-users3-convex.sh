#!/bin/bash
python solve.py --year=2012 --output=real1-users3-convex --lr=1e-5 --train-gt=lmdb-real1_users3_convex-gt3 --expectation --location --objectness --constraint --classes --display=1 --gpu=2
