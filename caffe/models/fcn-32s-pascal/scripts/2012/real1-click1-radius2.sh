#!/bin/bash
python solve.py --year=2012 --output=real1-click1-radius2 --lr=1e-5 --train-gt=lmdb-real1_click1_radius2-gt3 --expectation --location --objectness --constraint --classes --gpu=2 --display=1 --init-from=/imagenetdb3/abearman/caffe/models/clean-fcn-32s-pascal/2012/real1-click1-radius2/iters/real1-click1-radius2_step12800.caffemodel
