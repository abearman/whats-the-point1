#!/bin/bash
python /imagenetdb3/abearman/caffe/models/code/solve.py --year=2012 --output=real1-click1-cls-con-obj-random-from-3-users-1 --lr=1e-5 --train-gt=/imagenetdb3/abearman/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-real1_click1_random_from_3users_1-gt3 --expectation --location --objectness --constraint --classes --gpu=2 --display=1 
