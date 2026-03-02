#!/bin/bash
# 3-task 快速验证: AerialMaritime(5cls) + CottontailRabbits(1cls) + Egohands(1cls)
export CUDA_VISIBLE_DEVICES=4,5

rm -rf output/nsps_3task
mkdir -p output/nsps_3task

exec /opt/data/private/conda_envs/dithub/bin/python -u run_nsps_3task.py 2>&1 | tee output/nsps_3task/log.txt
