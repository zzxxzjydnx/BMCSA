#!/bin/sh
CURDIR="`dirname $0`" #获取此脚本所在目录
echo $CURDIR
cd $CURDIR #切换到该脚本所在目录

CUDA_VISIBLE_DEVICES=0 python train.py
CUDA_VISIBLE_DEVICES=2 python train-42.py
CUDA_VISIBLE_DEVICES=0 python train-ran.py
