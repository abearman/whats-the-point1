#!/bin/bash
python solve.py --year=2012 --output=siftflow-fs --lr=1e-10 --siftflow --train-gt=lmdb-fs-gt1 --val-gt=lmdb-fs-gt1 --display=1 --gpu=0
