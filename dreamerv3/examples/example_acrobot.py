def main():

    import warnings
    import dreamerv3
    from dreamerv3 import embodied
    import gym
    import time
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')  # Ignoring certain types of warnings

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])  # Loading default configuration
    config = config.update(dreamerv3.configs['medium'])  # Updating with medium configuration
    config = config.update({  # Further customizing configuration
        'logdir': f'logdir/{int(time.time())}',  # Directory for logs. Created fresh each time
        'run.train_ratio': 64,
        'run.log_every': 30,  # Logging frequency (in seconds)
        'batch_size': 16,  # Batch size for training
        'jax.prealloc': False,
        'encoder.mlp_keys': '.*',
        'decoder.mlp_keys': '.*',
        'encoder.cnn_keys': '$^',
        'decoder.cnn_keys': '$^',
        # 'jax.platform': 'cpu',
    })
    config = embodied.Flags(config).parse()  # Parsing final configuration

    logdir = embodied.Path(config.logdir)  # Path to log directory
    step = embodied.Counter()  # Step counter
    # Logger for different output formats
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),  # Logging to terminal
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),  # Logging metrics in JSONL format
        embodied.logger.TensorBoardOutput(logdir),  # TensorBoard output
        # embodied.logger.WandBOutput(logdir.name, config),  # WandB output (commented out)
        # embodied.logger.MLFlowOutput(logdir.name),  # MLFlow output (commented out)
    ])

    from embodied.envs import from_gym   
    env = gym.make("Acrobot-v1")

    env = from_gym.FromGym(env, obs_key='state_vec')  # Wrapping gym environment to use state vector as observation
    env = dreamerv3.wrap_env(env, config)  # Wrapping the environment with DreamerV3 specifics
    env = embodied.BatchEnv([env], parallel=False)  # Creating a batched version of the environment

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)  # Creating DreamerV3 agent
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay')  # Replay buffer for experience replay
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)  # Arguments for training
    # embodied.run.train(agent, env, replay, logger, args)  # Starting the training process
    # embodied.run.eval_only(agent, env, logger, args)  # Evaluation mode
    embodied.run.train_eval(agent, env, env, replay, replay, logger, args)  # Training and evaluation mode

if __name__ == '__main__':
    main()
