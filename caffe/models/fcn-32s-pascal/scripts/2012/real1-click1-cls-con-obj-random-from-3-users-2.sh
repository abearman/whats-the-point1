#!/bin/bash
python /imagenetdb3/abearman/caffe/models/code/solve.py --year=2012 --output=real1-click1-cls-con-obj-random-from-3-users-2 --lr=1e-5 --train-gt=/imagenetdb3/abearman/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-real1_click1_random_from_3users_2-gt3 --expectation --location --objectness --constraint --classes --gpu=2 --display=1 --init-from=/imagenetdb3/abearman/caffe/models/fcn-32s-pascal/2012/real1-click1-cls-con-obj-random-from-3-users-2/iters/real1-click1-cls-con-obj-random-from-3-users-2_step25000.caffemodel --start-iter=25000
 
