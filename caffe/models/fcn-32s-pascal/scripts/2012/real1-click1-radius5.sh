#!/bin/bash
python solve.py --year=2012 --output=real1-click1-radius5 --lr=1e-5 --train-gt=lmdb-real1_click1_radius5-gt3 --expectation --location --objectness --constraint --classes --gpu=3 --display=1 
