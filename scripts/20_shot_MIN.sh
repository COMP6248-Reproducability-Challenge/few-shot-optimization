#!/usr/bin/env bash
python3 -u train_meta_learner.py \
    -its 50000\
    -cuda 1 \
    -val 1000 \
    -data MIN \
    -classes 10 \
    -data_root data/miniImagenet/ \
    -save_state MIN20class20shot \
    -shots 20
