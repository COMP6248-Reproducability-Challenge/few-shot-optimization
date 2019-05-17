#!/usr/bin/env bash
python3 -u train_meta_learner.py \
    -its 10000\
    -cuda 0 \
    -val 1000 \
    -data MIN \
    -classes 10 \
    -data_root data/miniImagenet/ \
    -channel gscale \
    -save_state mings10class10shot \
    -shots 10
