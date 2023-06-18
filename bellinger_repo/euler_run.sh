#!/bin/bash

#SBATCH --gpus=gtx_1080_ti:1
#SBATCH --time=0-4

# conda activate bellinger
module load gcc glew eth_proxy cuda cudnn
python wrapper_demo.py
