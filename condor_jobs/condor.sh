#!/bin/bash

export PYTHONPATH=.

MY_SUPER_FLAGS="mode=FAST_RUN,device=gpu,floatX=float32"
/home/ehoang/miniconda3/envs/py37/bin/python /home/ehoang/hnt/tweet_summarization/models/main.py
