#!/bin/bash
python /imagenetdb3/abearman/caffe/models/code/solve.py --year=2012 --output=real1-click1-cls --lr=1e-5 --train-gt=lmdb-real1_click1-gt3 --expectation --location --classes --display=1 --gpu=1
