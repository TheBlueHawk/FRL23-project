#!/bin/bash

#SBATCH --gpus=gtx_1080_ti:1
#SBATCH --time=0-4

module load gcc/4.8.5 cuda/10.0.130 cudnn/7.4 eth_proxy
conda activate pets
python scripts/mbexp.py -env cartpole
