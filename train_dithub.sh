#!/bin/bash
# Initialize conda
source /root/anaconda3/etc/profile.d/conda.sh
conda activate dithub

# Set environment variables
export LD_LIBRARY_PATH=/opt/data/private/conda_envs/dithub/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1

python -u main.py --config-file test/test_odinw13 --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_dithub.py --model-checkpoint-path groundingdino_swint_ogc.pth --output-dir ./output/dithub_output --seed 3 --num-gpus 4 --shuffle-tasks --dithub --train-only