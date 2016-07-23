#!/bin/bash
python solve.py --year=2011 --fs-no-bg-constraint --lr=1e-10 --train-gt=lmdb-fs-minus-bg-gt3 --expectation --location --constraint --gpu=3 
