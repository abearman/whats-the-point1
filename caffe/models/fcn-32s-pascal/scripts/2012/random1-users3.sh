#!/bin/bash
python solve.py --year=2012 --output=random1-users3 --lr=1e-5 --train-gt=lmdb-random1_users3-gt3 --expectation --location --objectness --constraint --classes --init-from=/imagenetdb3/abearman/caffe/models/clean-fcn-32s-pascal/2012/random1-users3/iters/random1-users3_step12800.caffemodel --display=1 --gpu=2
