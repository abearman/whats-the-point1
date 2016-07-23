#!/bin/bash
python solve.py --year=2012 --output=bbox-just-supervised --lr=1e-10 --train-gt=lmdb-bbox-gt3 --expectation --location --no_norm_sup --no_norm_cls --no_norm_con  --display=1 --gpu=0 
