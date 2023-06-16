# COPY AND RUN LINE BY LINE
# srun --gpus=1 --ntasks=4 --mem-per-cpu=4G --pty bash
# conda create -n pets python=3.6
# conda activate pets
# pip install -r requirements.txt
# module load gcc/4.8.5 cuda/10.0.130 cudnn/7.4 eth_proxy
# python scripts/mbexp.py -env cartpole
