#!/usr/bin/env bash
python3 -u train_meta_learner.py \
    -its 10000\
    -cuda 0 \
    -val 1000 \
    -test \
    -data MNIST \
    -classes 10 \
    -data_root ../data/ \
    -load_state mings10class10shot \
    -channel gscale \
    -shots 10