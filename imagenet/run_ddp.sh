#!/bin/bash -a

CUDA_VISIBLE_DEVICES=0
MASTER_ADDR=$1
MASTER_PORT=$2
WORLD_SIZE=$3
RANK=$4
python -u main.py -a resnet50  --dist-backend 'gloo' --dist-url 'env://' --multiprocessing-distributed -b 64 --gpu 0 /mnt/nvme/datasets/imagenet/
