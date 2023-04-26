# Deep-Variational-Reinforcement-Learning (for FoRL project modified version)
This is the code accompanying the paper [Deep Variational Reinforcement Learning for POMDPs](https://arxiv.org/abs/1806.02426) by Maximilian Igl, Luisa Zintgraf, Tuan Anh Le, Frank Wood and Shimon Whiteson.

Disclaimer: I change the baselines libuary of the original DVRL repo to stable-baselines3, and modify some parts to enable training with more recent libuaries (python3.8, pytorch 1.13.1, etc.)

# Running the code 


### Installing dependencies (for FoRL project modified version)
You will need
- Python v3.8.10 (I used [Anaconda](https://conda.io/docs/user-guide/install/index.html) but it should work with other distributions as well)

Installation:

1. Open a terminal under the same directory as this file.
    
2. Create a virtual environment named FoRL (or something you like).

    `virtualenv dvrl`

3. Activate the environment.

    `source dvrl/bin/activate`

    If you are using Windows, the command is

    `dvrl\Scripts\activate.bat`

4. Install required packages (adapted for the recent packages).

    `pip install -r requirements.txt`

5. Install stable-baselines3

    `pip install stable-baselines3[extra]`


in the main folder.

If you're running into an error with matplotlib on MacOS when running the RNN on MountainHike, you can use [this simple solution](https://stackoverflow.com/a/21789908/3730984).

### Running

From the main folder execute

```
python3 ./code/main.py -p with environment.config_file=openaiEnv.yaml
```
The results will be saved in the `saved_runs` folder in subfolders with incrementing numbers.

# Plotting

I included a very simple plotting script in the main folder:
```
python3 plot.py --id <id> [--metric <metric>]
```
where `<id>` is the ID of the experiment (created automatically and printed to command line when each run is started).
`<metric>` is which metric you want to plot. `result.true` is the default and probably what you want, i.e. the unclipped reward.

We use [sacred](https://github.com/IDSIA/sacred) for configuration and saving of results. It fully supports a more elaborat setup with SQL or noSQL databases in the background for storing and retrieving results. I stripped that functionality out for the release for ease of use but can highly recommend using it when working more extensively with the code.


# Reproducing results

Below are the commands to reproduce the results in the paper. Plots in the paper are averaged over 5 random seeds, but individual runs should qualitatively show the same results as training was fairly stable. If you run into problems, let me know (maximilian.igl@gmail.com).

## Default configuration

The default configuration can be found in `code/conf/` in the `default.yaml`. 
The environment must be specified in the command line by `environment.config_file='<envName>.yaml'`. The corresponding yaml file will be loaded as well (and overwrites some values in `default.yaml`, like for example the encoder/decoder architecture to match the observations space). 
Everything specified additionally in the command line overwrites the values in both yaml files.

DVRL:
```
python3 ./code/main.py -p with environment.config_file=openaiEnv.yaml environment.name=PongNoFrameskip-v0 algorithm.use_particle_filter=True algorithm.model.h_dim=256 algorithm.multiplier_backprop_length=10 algorithm.particle_filter.num_particles=15 opt.lr=2.0e-04 loss_function.encoding_loss_coef=0.1
```

RNN:
```
python3 ./code/main.py -p with environment.config_file=openaiEnv.yaml environment.name=PongNoFrameskip-v0 algorithm.use_particle_filter=False algorithm.model.h_dim=256 algorithm.multiplier_backprop_length=10  opt.lr=1.0e-04
```
(or with any other Atari environment)
**Please note that the results printed in the console are the _clipped_ rewards, for the true rewards please check 'result.true' in the metrics.json file or use the plotting script**

# Credits

The code is based on an older version of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr but heavily modified.


