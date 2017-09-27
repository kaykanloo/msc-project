#!/bin/sh
export PATH="/nobackup/sc16mk/miniconda3/bin:$PATH"
export PATH="/nobackup/sc16mk/cuda-8.0/bin:$PATH"
export LD_LIBRARY_PATH="/nobackup/sc16mk/cuda-8.0/lib64:$LD_LIBRARY_PATH"

python run.py training   Baseline
python run.py prediction Baseline
python run.py evaluation Baseline








