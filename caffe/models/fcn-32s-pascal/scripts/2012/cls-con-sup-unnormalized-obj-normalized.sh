#!/bin/bash
python solve.py --year=2012 --output=cls-con-sup-unnormalized-obj-normalized --lr=1e-10 --train-gt=lmdb-real1_click1-gt3 --expectation --location --objectness --constraint --no_norm_sup --no_norm_cls --no_norm_con --display=1 --gpu=0 
