#!/bin/bash
python solve.py --year=2012 --output=image-level-labels-objectness --lr=1e-5 --train-gt=lmdb-bbox-gt3 --expectation --objectness --gpu=1 --display=1 --momentum=0.9

