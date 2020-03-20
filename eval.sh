#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=2,3 python3 eval.py
# python3 eval.py --resume
python3 eval.py -net ir_se -depth 100
