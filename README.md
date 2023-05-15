# FRL23-project
## Benchmarking and improvement of ACNO-MDP’s


## Install and run
### Usual setup (every time)
On first login follow [euler-first-time-setup](#euler-first-time-setup). Otherwise:
```bash
ssh euler
# Open project and load most recent changes
cd ../../scratch/<user_name>/FRL23-project/dreamerv3
git pull
# Request interactive session with GPU (for quick exprimenting, default time-out 1H)
srun --gpus=1 --ntasks=4 --mem-per-cpu=4G --pty bash
# Load gpu modules, internet access, ...
module load gcc/8.2.0 python_gpu/3.10.4 r/4.0.2 git-lfs/2.3.0 eth_proxy npm/6.14.9 ffmpeg/5.0 cudnn/8.8.1.3 cuda/11.7.0 
# Activate environment 
conda activate dreamjax

# Run
python example.py
```


### Useful commands

#### Manage modules
- `module spider <keyword>`   Explore available modules 
- `module load <your_module>` Install your module, can concat several ones separated by space
- `module list`               List installed modules

#### Manage runs / jobs
- `squeue`	View job and job step information for jobs managed by Slurm
- `scontrol`	Display information about the resource usage of a job
- `sstat`	    Display the status information of a running job/step
- `sacct`	    Displays accounting data for all jobs and job steps in the Slurm job accounting log or Slurm database
- `myjobs`	Job information in human readable format
- `scancel`	Kill a job

#### Graphical output
- `xvfb-run -d <your_python_cmd>` Use a frame buffer for offscreen rendering
- `ssh -Y euler` Tunnel the graphical output to your local machine. Need to have a X11 server installed.


### Euler first time setup
#### SSH
- Generate ssh key
- Add to the `.ssh/config` file 
  
```bash
Host ethjump
  HostName jumphost.inf.ethz.ch
  User <nethz_username>
  PreferredAuthentications publickey
  IdentityFile ~/.ssh/<your_ssh_key>

Host euler
    HostName euler.ethz.ch
    User <nethz_username>
    ProxyJump ethjump
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/<your_ssh_key>
```

- Connect and save your ssh key, fill your password when prompted (once for the jumphost, once for euler)
```
ssh-copy-id -i ~/.ssh/<your_ssh_key>.pub euler
```
Now you can login on euler with `ssh euler`, no password or VPN requried!


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
# (you may need to run the following if pip does not work on Euler properly:)
# curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# python3 get-pip.py --force-reinstall

# Install TF
pip install tensorflow
# Install other requirements 
pip install -r requirements.txt
```
