#!/bin/bash
python /imagenetdb3/abearman/caffe/models/code/solve.py --year=2012 --output=real1-click1-cls-con-obj --lr=1e-5 --train-gt=/imagenetdb3/olga/data//segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-real1_click1-gt3 --expectation --location --objectness --constraint --classes --gpu=2 --display=1 
