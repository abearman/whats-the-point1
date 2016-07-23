#!/bin/bash
python solve.py --year=2012 --output=fs-cls-con --lr=1e-5 --train-img=/imagenetdb3/abearman/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-fs-10-percent-img/ --train-gt=/imagenetdb3/abearman/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-fs-10-percent-gt3/ --expectation --location --constraint --classes --display=1 --gpu=2

