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
python test.py 20 
python test.py 40 
python test.py 80 
python test.py 160 
python gradient.py 
