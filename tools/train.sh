#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 
