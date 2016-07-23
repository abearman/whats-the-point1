#!/bin/bash
#python solve.py --output=objects-no-bg-objectness --train-gt=lmdb-pascal_2011t-gt-ignorebg-plus-objectness --objectness=true

#python solve.py --output=objects-no-bg-objectness --train-gt=lmdb-pascal_2011t-gt-plus-objectness --objectness=true

python solve.py --year=2011 --output=user-clicks-objectness --train-gt=lmdb-pascal_2011t-real1-gt-plus-objectness --objectness --gpu=3 --lr=1e-10

#python solve.py --output=objects-no-bg-objectness --train-gt=lmdb-pascal_2011t-ignorebg-gt --objectness=true
