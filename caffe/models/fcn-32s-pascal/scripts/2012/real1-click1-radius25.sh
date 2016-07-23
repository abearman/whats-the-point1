#!/bin/bash
python solve.py --year=2012 --output=real1-click1-radius25 --lr=1e-5 --train-gt=lmdb-real1_click1_radius25-gt3 --expectation --location --objectness --constraint --classes --gpu=1 --display=1 
