#!/usr/bin/env bash
source ~/venv/bin/activate
python test.py 20 3
python test.py 40 3 
python test.py 80 3
python test.py 160 3 
python gradient.py 3

python test.py 20 5
python test.py 40 5 
python test.py 80 5
python test.py 160 5 
python gradient.py 5

python test.py 20 10
python test.py 40 10 
python test.py 80 10
python test.py 160 10 
python gradient.py 10