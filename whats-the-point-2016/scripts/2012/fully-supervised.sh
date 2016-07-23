#!/bin/bash
python solve.py --year=2012 --output=fully-supervised --lr=1e-10 --train-gt=/imagenetdb3/olga/data/segm_lmdb/lmdb-pascal_2012t_SBDtv_minus_2012v/lmdb-fs-gt1 --gpu=0

