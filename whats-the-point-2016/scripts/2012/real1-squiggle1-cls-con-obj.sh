#!/bin/bash
python /imagenetdb3/abearman/caffe/models/code/solve.py --year=2012 --output=real1-squiggle1-cls-con-obj --lr=1e-5 --train-gt=/imagenetdb3/abearman/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-real1_squiggle1-gt3 --expectation --location --objectness --constraint --classes --gpu=2 --display=1 --init-from=/imagenetdb3/abearman/caffe/models/fcn-32s-pascal/2012/real1-squiggle1-cls-con-obj/iters/real1-squiggle1-cls-con-obj_step33000.caffemodel --start-iter=33000

