#!/bin/bash
python /imagenetdb3/abearman/caffe/models/code/solve.py --year=2012 --output=image-level-labels-proportional-half --lr=1e-5 --train-img=/imagenetdb3/abearman/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-ill-proportional-half-img --train-gt=/imagenetdb3/abearman/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-ill-proportional-half-gt3 --expectation --constraint --gpu=1 --display=1 

#lmdb-bbox-gt3 
