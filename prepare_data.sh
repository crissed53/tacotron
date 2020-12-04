#!/bin/bash

set -exu

if [ $# -eq 0 ]
  then
    NUM_WORKER=2
  else
    NUM_WORKER=$1
fi

DATASET_DIR="./dataset/LJSpeech-1.1.tar.bz2"

if [ ! -d $DATASET_DIR ]
  then
    ./download_dataset.sh
fi

python3 ./gen_dataset.py --num_workers "$NUM_WORKER"
