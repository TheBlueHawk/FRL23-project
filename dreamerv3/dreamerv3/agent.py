import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
# Define a helper lambda function to stop the gradient flow during backpropagation.
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()

# Define a custom filter class for the logging.
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    # If the log message contains 'check_types', do not log it.
    return 'check_types' not in record.getMessage()
# Add the filter to the logger.
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj


@jaxagent.Wrapper  # Use the Wrapper decorator from jaxagent
class Agent(nj.Module):
  # Load the configuration file
  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    # Initialize the agent with observation space, action space, step and configuration
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    # Initialize the world model
    self.wm = WorldModel(obs_space, act_space, config, name='wm')
    # Determine the task behavior
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    # Determine the exploration behavior
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    # This method is used to initialize the policy's states.
    # It returns the initial states of the world model, the task behavior, and the exploration behavior.
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    # This method is used to initialize the training's states.
    # It returns the initial state of the world model.
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    # This method defines the policy of the agent based on the current state and the observations.
    # Depending on the mode ('eval', 'explore', or 'train'), it uses either the task or exploration behavior.

    # If the configuration specifies to use JIT (Just-In-Time) compilation, log a message.
    self.config.jax.jit and print('Tracing policy function.')
    
    # Preprocess the observations.
    obs = self.preprocess(obs)
    
    # Unpack the previous states.
    (prev_latent, prev_action), task_state, expl_state = state
    
    # Encode the observations using the world model's encoder.
    embed = self.wm.encoder(obs)

    latent, _ = self.wm.rssm.obs_step(prev_latent, prev_action, embed, obs['is_first']) # post = (z,h)
    
    # Execute the exploration behavior's policy, given the current latent state.
    self.expl_behavior.policy(latent, expl_state)
    
    task_outs, task_state = self.task_behavior.policy(latent, task_state) # actor policy, task_state just a "carry"
    
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state) # for us: same as task_behavior
    
    # Depending on the mode, choose which policy to use and what action to take.
    if mode == 'eval':
      outs = task_outs
      outs['action'] = outs['action'].sample(seed=nj.rng())
      outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
    elif mode == 'explore':
      outs = expl_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'train':
      outs = task_outs # actor output = distribution on actions
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng()) # returns entropy of action and 1 sample of it
    
    # Update the states.
    state = ((latent, outs['action']), task_state, expl_state)
    
    # Return the policy outputs and the updated state.
    return outs, state # = (actions_sampled, actions_entropy), (((z,h), action_sampled), {}, {})

  def train(self, data, state, imaginary):
    # This method is used to train the agent using the given data and state.

    # abstract_value = jnp.array(True, dtype=bool)  # Abstract tracer value
    # concrete_value = abstract_value.astype(bool)  # Convert to concrete boolean

    # print("abstract: ", abstract_value)
    # print("concrete: ", concrete_value)

    # print("imaginary: ", imaginary)
    # print("imaginary type int: ", imaginary.astype(int))

    # x = "this is a test"

    # def true_func():
    #     print(x)

    # def false_func():
    #     print("FALSE_FUNCTION IS PRINTED")
  

    # jax.lax.cond(jnp.equal(imaginary, 1), true_func, false_func)



    # If the configuration specifies to use JIT (Just-In-Time) compilation, log a message.
    self.config.jax.jit and print('Tracing train function.')

    # Initialize an empty dictionary to store metrics.
    metrics = {}

    # Preprocess the data.
    data = self.preprocess(data)

    # Train the world model using the data and update the state.
    # Obtain the world model's outputs and the metrics.

    # def dummy_function():
    #   return None, None, None

    # state, wm_outs, mets = jax.lax.cond(jnp.equal(imaginary, 0), lambda: self.wm.train(data, state), dummy_function)

    state, wm_outs, mets = self.wm.train(data, state)

    # state, wm_outs, mets = jax.lax.cond(jnp.equal(imaginary, 0), lambda: self.wm.train(data, state), lambda: self.wm.fake_train_wm_outs(data, state))

    # Update the metrics dictionary with the metrics from the world model.
    metrics.update(mets)

    # Combine the data and the outputs of the world model to form the context for the behaviors.
    context = {**data, **wm_outs['post']}

    # Modify the shape of the context.
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)

    # Train the task behavior using the world model's imagine method, the start state and the context.
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)

    policy = lambda s: self.task_behavior.ac.actor(sg(s)).sample(seed=nj.rng())
    traj = self.wm.imagine(policy, start, 1)

    # Update the metrics dictionary with the metrics from the task behavior.
    metrics.update(mets)

    # If the exploration behavior is configured, train it and update the metrics dictionary.
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})

    # Initialize an empty dictionary to store outputs.
    outs = {}

    # Return the outputs, the updated state, and the metrics.
    return outs, state, metrics, traj

  def report(self, data):
    # This method is used to report the metrics of the agent's behaviors.

    # If the configuration specifies to use JIT (Just-In-Time) compilation, log a message.
    self.config.jax.jit and print('Tracing report function.')

    # Preprocess the data.
    data = self.preprocess(data)

    # Initialize an empty dictionary to store the report.
    report = {}

    # Update the report with the world model's report.
    report.update(self.wm.report(data))

    # Get the report from the task behavior and update the report dictionary.
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})

    # If the exploration behavior is not the same as the task behavior, get its report and update the report dictionary.
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})

    # Return the report.
    return report

  def preprocess(self, obs):
    # This method is used to preprocess the observations.
    
    # Make a copy of the observations.
    obs = obs.copy()

    # Loop over each item in the observations dictionary.
    for key, value in obs.items():
      
      # Skip the items with keys starting with 'log_' or the key 'key'.
      if key.startswith('log_') or key in ('key',):
        continue
      
      # If the value is an image, normalize it.
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)

      # Update the value in the observations dictionary.
      obs[key] = value

    # Add a new item 'cont' to the dictionary, which is 1.0 where 'is_terminal' is 0.0, and vice versa.
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)

    # Return the preprocessed observations.
    return obs


class WorldModel(nj.Module):
  # The WorldModel class inherits from the nj.Module class. It represents the agent's model of the world.

  def __init__(self, obs_space, act_space, config):
    # The constructor of the WorldModel class. It initializes the world model with the given observation space, 
    # action space, and configuration.

    # Store the observation space, action space, and configuration.
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config

    # Create a dictionary of shapes of the observations. Exclude those keys which start with 'log_'.
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}

    # Initialize an encoder, RSSM (Recurrent Stochastic State Model), decoder, reward head, and continuation head.
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    self.rssm = nets.RSSM(**config.rssm, name='rssm')
    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont')}

    # Initialize an optimizer.
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    self.fake_opt = jaxutils.Optimizer(name='model_opt', **config.model_fake_opt) 

    # Create a dictionary of loss scales. Use different scales for image and vector data.
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    # This method returns the initial latent state and action for the given batch size.

    # Get the initial latent state from the RSSM.
    prev_latent = self.rssm.initial(batch_size)

    # Initialize the previous action as zeros.
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))

    return prev_latent, prev_action

  def train(self, data, state):
    # This method trains the world model using the given data and state.

    # Create a list of modules to train.
    modules = [self.encoder, self.rssm, *self.heads.values()]

    # Train the modules using the optimizer. The loss function is defined elsewhere in the code.
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)

    # Update the metrics with the metrics from the optimizer.
    metrics.update(mets)

    return state, outs, metrics
  

  def fake_train(self, data, state):
    # This method trains the world model using the given data and state.

    # Create a list of modules to train.
    modules = [self.encoder, self.rssm, *self.heads.values()]

    # Train the modules using the optimizer. The loss function is defined elsewhere in the code.
    mets, (state, outs, metrics) = self.fake_opt(
        modules, self.loss, data, state, has_aux=True)

    # Update the metrics with the metrics from the optimizer.
    metrics.update(mets)

    return state, outs, metrics


  def loss(self, data, state):
    # This method calculates the loss for training the world model.

    # Encode the input data.
    embed = self.encoder(data)

    # Split the state into previous latent state and previous action.
    prev_latent, prev_action = state

    # Concatenate previous actions with the current actions (excluding the last action in the data).
    prev_actions = jnp.concatenate([prev_action[:, None], data['action'][:, :-1]], 1)

    # Apply the observation model in the RSSM to get the posterior and prior distributions over latent states.
    post, prior = self.rssm.observe(embed, prev_actions, data['is_first'], prev_latent)

    # Initialize an empty dictionary to store the distributions of different heads.
    dists = {}

    # Combine the posterior distributions and embeddings into one dictionary.
    feats = {**post, 'embed': embed}

    # For each head, get the distribution and add it to the dists dictionary.
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)

    # Initialize an empty dictionary to store the losses.
    losses = {}

    # Calculate the dynamics loss and representation loss using the RSSM.
    losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)

    # For each distribution, calculate the negative log-likelihood as the loss and add it to the losses dictionary.
    for key, dist in dists.items():
      loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss

    # Scale the losses according to the scales defined in the model.
    scaled = {k: v * self.scales[k] for k, v in losses.items()}

    # Sum all scaled losses to get the total model loss.
    model_loss = sum(scaled.values())

    # Prepare output data that includes embeddings, posterior and prior distributions, and losses.
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})

    # Get the last latent state and action for the next training step.
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = data['action'][:, -1]
    state = last_latent, last_action

    # Calculate the metrics for monitoring.
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)

    # Return the mean model loss, the new state, output data, and metrics.
    return model_loss.mean(), (state, out, metrics)

  def imagine(self, policy, start, horizon):
    # This method generates an imagined trajectory using the given policy.

    # Get the first continuation signal.
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)

    # Filter the start state to only include the keys present in the initial state of the RSSM.
    keys = list(self.rssm.initial(1).keys())
    start = {k: v for k, v in start.items() if k in keys}

    # Get the initial action using the policy.
    start['action'] = policy(start)

    # Define a step function to propagate the state in the imagined trajectory.
    def step(prev, _):
      prev = prev.copy()
      state = self.rssm.img_step(prev, prev.pop('action'))
      return {**state, 'action': policy(state)}

    # Use the scan function to iterate the step function over the horizon to generate an imagined trajectory.
    traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)

    # Concatenate the start state with the imagined trajectory.
    traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}

    # Get the continuation predictions for the imagined trajectory.
    cont = self.heads['cont'](traj).mode()

    # Concatenate the first continuation signal with the rest of the continuation predictions.
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)

    # Calculate the weight for each step in the trajectory based on the continuation predictions.
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount

    # Return the imagined trajectory.
    return traj

  def report(self, data):
    # This method generates a report of the model's performance, useful for debugging and monitoring.

    # Initialize the model's state.
    state = self.initial(len(data['is_first']))
    
    # Initialize an empty dictionary to store the report.
    report = {}

    # Update the report with the metrics from the loss method.
    report.update(self.loss(data, state)[-1][-1])

    # Apply the observation model in the RSSM to get the context, using a subset of the data.
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])

    # Get the last latent state for each feature in the context.
    start = {k: v[:, -1] for k, v in context.items()}

    # Use the decoder head to reconstruct the features from the context.
    recon = self.heads['decoder'](context)

    # Use the decoder head to generate a trajectory by imagining the future actions.
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))

    # For each feature, calculate the error between the true and predicted values and store them in the report.
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)

    # Return the report.
    return report

  def _metrics(self, data, dists, post, prior, losses, model_loss):
    # This private method calculates various metrics for monitoring the model's performance.

    # Define a function to compute entropy.
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()

    # Initialize an empty dictionary to store the metrics.
    metrics = {}

    # Calculate entropy of the prior and posterior distributions, and add them to the metrics.
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))

    # Calculate mean and standard deviation of the losses for each key in losses, and add them to the metrics.
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})

    # Calculate the mean and standard deviation of the overall model loss, and add them to the metrics.
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()

    # Calculate the maximum absolute reward in the data and the prediction, and add them to the metrics.
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()

    # If 'reward' is present in dists and there are no NaNs in the data,
    # calculate balance statistics for the reward distribution and add them to the metrics.
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})

    # If 'cont' is present in dists and there are no NaNs in the data,
    # calculate balance statistics for the 'cont' distribution and add them to the metrics.
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})

    # Return the computed metrics.
    return metrics

class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    # Initializing the module and configuring the critics based on the scales (weights).
    # We're only including critics for which the scale (weight) is non-zero.
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales

    # Initializing the action space and configuration
    self.act_space = act_space
    self.config = config

    # Setting up the gradient strategy and actor network
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)

    # Initializing moments for return normalization
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}

    # Initializing the optimizer for the actor
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    # A placeholder method that returns an empty dictionary. This can be used to initialize some state before training.
    return {}

  def policy(self, state, carry):
    # This method defines the policy of the actor. The actor network is used to decide the action to take.
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context):
  # The train method is responsible for updating the parameters of the actor and the critics based on the given start state and context.
    def loss(start):
      # In this nested function, a policy function is defined using the actor network.
      # The imagine function is used to generate a trajectory following this policy for a certain horizon.
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      
      # The loss and metrics for the actor are computed based on this trajectory.
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    
    #TODO: change this function not to learn when set to imaginary !
    
    # The optimizer updates the parameters of the actor to minimize the loss.
    # The updated metrics from the optimizer and from each critic's training are collected.
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
      
    # The method returns the trajectory and the collected metrics.
    return traj, metrics

  def loss(self, traj):
    # The loss method computes the loss for the actor based on a given trajectory.
    # The loss is calculated based on the advantages computed for each critic.
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      # For each critic, a score is computed for the trajectory.
      rew, ret, base = critic.score(traj, self.actor)
      
      # The returns are normalized based on previously computed moments.
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      
      # The advantage for this critic is calculated as the scaled difference between the normalized return and baseline.
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      
      # Some statistics are computed for the rewards and returns.
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
      
    # The advantages are aggregated to compute the loss for the actor.
    # The loss is computed differently depending on the chosen gradient strategy (either backpropagation or reinforce).
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    
    # The entropy regularization term is added to the loss.
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    
    # The loss is scaled by the weight of the trajectory and a loss scale factor.
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    
    # The metrics are updated with metrics computed from the trajectory, policy, log probabilities, entropy, and advantage.
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    
    # The method returns the mean loss and the metrics.
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    # This method computes various metrics for evaluating the performance of the actor and the policy.
    metrics = {}
    
    # The entropy, randomness, and log probability of the policy are computed.
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    
    # The metrics are updated with statistics computed from the actions, policy randomness, policy entropy, log probabilities, and advantage.
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    
    # The distribution of weights in the trajectory is also included in the metrics.
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    
    # The method returns the computed metrics.
    return metrics

class VFunction(nj.Module):
  def __init__(self, rewfn, config):
    # In the constructor, a reward function and configuration are taken as inputs.
    # Two MLP networks (net and slow) are initialized for the critic, and an optimizer for the critic and an updater for the slow network are created.
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, actor):
    # The train method is responsible for updating the parameters of the critic based on the given trajectory.
    # The target for the training is the score of the trajectory.
    target = sg(self.score(traj)[1])
    
    # The optimizer updates the parameters of the critic to minimize the loss.
    # The updated metrics from the optimizer are collected.
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    
    # The parameters of the slow network are updated.
    self.updater()
    
    # The method returns the collected metrics.
    return metrics

  def loss(self, traj, target):
    # The loss method computes the loss for the critic based on a given trajectory and target.
    # The trajectory is trimmed to remove the final state.
    metrics = {} 
    traj = {k: v[:-1] for k, v in traj.items()}
    
    # The network computes a distribution for the trajectory.
    # The loss is computed as the negative log likelihood of the target under this distribution.
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    
    # Depending on the chosen regularization strategy, an additional regularization term is computed.
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    
    # The regularization term is added to the loss.
    loss += self.config.loss_scales.slowreg * reg
    
    # The loss is scaled by the weight of the trajectory and a loss scale factor.
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    
    # Some statistics are computed for the mean of the critic's distribution.
    metrics = jaxutils.tensorstats(dist.mean())
    
    # The method returns the loss and the metrics.
    return loss, metrics

  def score(self, traj, actor=None):
    # The score method computes a reward, return, and baseline for a given trajectory.
    # The reward is computed using the provided reward function.
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    
    # The discount factor is computed based on the continuous trajectory and the horizon from the configuration.
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    
    # The value is computed using the critic's network.
    value = self.net(traj).mean()
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    
    # The return is computed by backtracking through the trajectory.
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    
    # The method returns the reward, return, and baseline.
    return rew, ret, value[:-1]