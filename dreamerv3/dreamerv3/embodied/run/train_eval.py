import re

import embodied
import numpy as np


def train_eval(
    agent, train_env, eval_env, train_replay, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()
  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({
        'length': length, 'score': score,
        'reward_rate': (ep['reward'] - ep['reward'].min() >= 0.1).mean(),
    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
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
    metrics.add(stats, prefix=f'{mode}_stats')

  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(train_replay.add)
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

  random_agent = embodied.RandomAgent(train_env.act_space)
  print('Prefill train dataset.')
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    driver_train(random_agent.policy, steps=100)
  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill): # for us, no loop as eval_replay = train_replay so already full
    driver_eval(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset_train = agent.dataset(train_replay.dataset)
  dataset_eval = agent.dataset(eval_replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    # create imaginary trajectory first:
    has_traj = False
    if batch[0] is not None and state[0] is not None:
      # new_dict_batch = {}
      # new_dict_state = {}
      # (state_), action = state[0]
      # print("shape of state[0] deter", state_['deter'].shape)
      # print("shape of state[0] logit", state_['logit'].shape)
      # print("shape of state[0] stoch", state_['stoch'].shape)
      # print("shape of state[0] action", action.shape)

      # for key, value in batch[0].items():
      #   new_dict_batch[key] = value[0:1, 0:1, :]  # Extract the first value and reshape to size [1, 1, x]
      # for key, value in state_.items():
      #   new_dict_state[key] = value[0:1, 0:1, :]  # Extract the first value and reshape to size [1, 1, x]
      # new_action = action[0:1,:]

      # new_state = (new_dict_state), new_action

      # print shape of batch[0] and state[0] for each key:
      for key, value in batch[0].items():
        print("shape of batch[0] ", key, value.shape)

      for key, value in state[0][0].items():
        print("shape of state[0] ", key, value.shape)

      print("shape of state[0] action", state[0][1].shape)

      _, _, _, traj = agent.train(batch[0], state[0], imaginary=1)
      # print("batch : ",batch[0])
      # print("state : ",state[0])
      has_traj = True
      print("inside train_step: ", traj["action"].shape)

    # regular function afterwards

    for _ in range(should_train(step)): # True once every 1/Ratio steps (default: 16)
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train) # dictionary with dimension [batch_size, batch_length, x] (x depends on the key)
      outs, state[0], mets, _ = agent.train(batch[0], state[0],  imaginary=0)
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        train_replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates): # some weird stuff to make several devices work together, for parallelization
      agent.sync()
    if should_log(step): # every x seconds (config log_every) => add data to logger and save / print the metrics (here that we print long summary)
      logger.add(metrics.result())
      logger.add(agent.report(batch[0]), prefix='report')
      with timer.scope('dataset_eval'):
        eval_batch = next(dataset_eval)
      logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay') # replay stats:  'size' ('inserts', 'samples', 'insert_wait_avg', 
      logger.add(eval_replay.stats, prefix='eval_replay')           # 'insert_wait_frac', 'sample_wait_avg', 'sample_wait_frac')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
    if has_traj:
      return traj
    return None
  driver_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    if should_eval(step): # eval every "eval_every" steps, and always the first time
      print('Starting evaluation at step', int(step))
      driver_eval.reset()
      driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps)) # for us: one full episode in eval
    driver_train(policy_train, steps=100)
    if should_save(step):
      checkpoint.save()
  logger.write()
  logger.write()
