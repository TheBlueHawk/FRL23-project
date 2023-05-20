import re
import embodied
import numpy as np

def train(agent, env, replay, logger, args):

  # Create log directory and print its path
  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)

  # Define conditions for exploration, training, logging, saving, and synchronization
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_sync = embodied.when.Every(args.sync_every)

  # Initialize step counter, updates counter, and metrics tracker
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()

  # Print observation space and action space information
  print('Observation space:', embodied.format(env.obs_space), sep='\n')
  print('Action space:', embodied.format(env.act_space), sep='\n')

  # Create timer object and wrap it around agent, environment, replay, and logger
  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])
  timer.wrap('replay', replay, ['add', 'save'])
  timer.wrap('logger', logger, ['write'])

  # Initialize a set to keep track of non-zero keys for logging purposes
  nonzeros = set()

  # Define a function to be called per episode for logging and reporting
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    sum_abs_reward = float(np.abs(ep['reward']).astype(np.float64).sum())

    # Log episode statistics
    logger.add({
        'length': length,
        'score': score,
        'sum_abs_reward': sum_abs_reward,
        'reward_rate': (np.abs(ep['reward']) >= 0.5).mean(),
    }, prefix='episode')

    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}

    # Process additional episode statistics for logging
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix='stats')

  # Create a driver object to control the training loop
  driver = embodied.Driver(env)

  # Attach the per_episode function to be called after each episode
  driver.on_episode(lambda ep, worker: per_episode(ep))

  # Attach logger.step.increment() function to be called after each step
  driver.on_step(lambda tran, _: step.increment())

  # Attach replay.add function to be called after each step
  driver.on_step(replay.add)

  print('Prefill train dataset.')

  # Step 1: Fill the replay buffer with transitions using a random agent
  random_agent = embodied.RandomAgent(env.act_space)
  while len(replay) < max(args.batch_steps, args.train_fill):
    driver(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  # Create a dataset from the replay buffer
  dataset = agent.dataset(replay.dataset)
  state = [None]  # To be writable from train step function below
  batch = [None]

  # Define the training step function
  def train_step(tran, worker):
    # Perform training steps according to the should_train condition
    for _ in range(should_train(step)):
      with timer.scope('dataset'):
        batch[0] = next(dataset)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')

      # Prioritize replay buffer if priorities are available in training outputs
      if 'priority' in outs:
        replay.prioritize(outs['key'], outs['priority'])

      updates.increment()

    # Synchronize the agent if the should_sync condition is met
    if should_sync(updates):
      agent.sync()

    # Log metrics and other information if the should_log condition is met
    if should_log(step):
      agg = metrics.result()
      report = agent.report(batch[0])
      report = {k: v for k, v in report.items() if 'train/' + k not in agg}
      logger.add(agg)
      logger.add(report, prefix='report')
      logger.add(replay.stats, prefix='replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)

  # Attach the train_step function to be called after each step
  driver.on_step(train_step)

  # Create a checkpoint object to save and load agent, replay buffer, and step information
  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  timer.wrap('checkpoint', checkpoint, ['save', 'load'])
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.replay = replay

  # Load from a checkpoint if specified, otherwise load or save from the latest checkpoint
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we just saved

  print('Start training loop.')

  # Define the policy function to be passed to the driver for action selection
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')

  # Run the training loop until the desired number of steps is reached
  while step < args.steps:
    driver(policy, steps=100)
    if should_save(step):
      checkpoint.save()
  logger.write()
