#!/bin/bash
python solve.py --year=2012 --output=real1-rankweight --lr=1e-5 --train-gt=lmdb-real1_rankweight-gt4 --expectation --location --objectness --constraint --classes --rank --display=1 --gpu=3
