#!/usr/bin/env bash
python3 -u train_meta_learner.py \
    -its 10000\
    -cuda 0 \
    -val 1000 \
    -data MIN \
    -classes 5 \
    -data_root data/miniImagenet/ \
    -save_state minstate1205 \
    -shots 5
