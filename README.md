# FRL23-project
## DreamerV3

#### Repo setup
After setting up the github credentials (email, username, token): 

```bash
# Clone project in scratch space 
cd ../../scratch/<user_name>/
git clone https://github.com/TheBlueHawk/FRL23-project.git

# Create and activate env
conda create --name dreamjax python=3.10
conda activate dreamjax

# Install JAX
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# You might need to update setuptools to:
pip install setuptools==57.1.0

# Install TF
pip install tensorflow

# Install other requirements 
pip install -r requirements.txt
```

## PETS
#### Create environment
``` bash
conda create -n pets python=3.6
conda activate pets
pip install -r requirements.txt
```

#### Run example
``` bash
srun --gpus=1 --ntasks=4 --mem-per-cpu=4G --pty bash
conda activate pets
module load gcc/4.8.5 cuda/10.0.130 cudnn/7.4 eth_proxy
python scripts/mbexp.py -env cartpole
```

#### Adjust approach
* To run Naive approach:
  * Set the number of training episodes in `dmbrl/misc/MBExp.py`, at line 99
  * Set the number of deployment episodes in `dmbrl/misc/MBExp.py`, at line 100
  * Set the number of imaginary steps between two observations in `dmbrl/misc/Agent.py`, at line 82
 
* To run Heuristics approach:
  * Set the number of training episodes in `dmbrl/misc/MBExp.py`, at line 99
  * Set the number of deployment episodes in `dmbrl/misc/MBExp.py`, at line 100
  * Deactivate the Naive approach by commenting the line 82 in `dmbrl/misc/Agent.py`
  * Set the variance threshold in `dmbrl/misc/Agent.py`, at line 79

