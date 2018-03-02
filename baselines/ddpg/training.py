import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory, 
    load_network_id, latest, plot_info,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    
    # logger.info('Using agent with the following configuration:')
    # logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        if (load_network_id):
            agent.load_actor_critic(id=load_network_id)
        if (latest):
            agent.load_actor_critic(latest=True)

        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_distances2target = []
        epoch_episode_relative_alt = []
        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        logger.warn('EPISODE OVER!')
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        epoch_episode_distances2target.append(info['dist2target'])
                        epoch_episode_relative_alt.append(info['relative_alt'])
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        if (episodes % 10 == 0):
                            agent.save_actor_critic(id=episodes)

                        if (episodes % 2 == 0 and plot_info):
                            plot_information(
                                      epoch_episode_distances2target, 
                                      epoch_episode_rewards, 
                                      epoch_episode_relative_alt)
                            plt.pause(0.1)
                        agent.reset()
                        obs = env.reset()

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.

            # Update learning rates
            if (epoch % 5 == 0 and epoch > 0):
                agent.update_lr(agent.actor_lr*0.65, agent.critic_lr*0.65)

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = stats.copy()
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = eval_episode_rewards
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = eval_qs
                combined_stats['eval/episodes'] = len(eval_episode_rewards)
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            logger.info('----------------------------------------')
            logger.info(combined_stats)
            
            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

def plot_information(dist, reward, alt):
    # Plot distance to target
    fig = plt.figure(1)
    ax1 = fig.add_subplot(311)
    count = len(dist)

    ax1.set_title('Distances to target at end of trajectory')    
    ax1.set_xlabel('Trajectory count')
    ax1.set_ylabel('Distance (m)')
    plt.grid(linestyle='-')
    plt.tight_layout()
    plt.xlim(xmin=0, xmax=len(dist))
    plt.ylim(ymin=0, ymax=max(dist))
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot horizontal line
    plt.axhline(y=17243.39, color='k', linewidth=2, alpha=0.5, label='Distance from home (17243m.)' if count <= 2 else '_nolabel_')
    plt.axhline(y=500., color='m', linewidth=2, alpha=0.5, label='Eps distance (500m.)' if count <= 2 else '_nolabel_')

    # Plot distances
    ax1.plot(dist, c='r', label='Distance (m)' if count <= 2 else '_nolabel_')
    leg = ax1.legend()

    # Plot rewards
    fig = plt.figure(1)
    ax1 = fig.add_subplot(312)

    ax1.set_title('Reward for a complete trajectory')    
    ax1.set_xlabel('Trajectory count')
    ax1.set_ylabel('Reward')
    plt.grid(linestyle='-')
    plt.tight_layout()
    plt.xlim(xmin=0, xmax=len(reward))
    plt.ylim(ymin=min(reward), ymax=max(reward))
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot rewards
    ax1.plot(reward, c='b', label='Reward' if count <= 2 else '_nolabel_')
    leg = ax1.legend()


    # Plot height
    fig = plt.figure(1)
    ax1 = fig.add_subplot(313)

    ax1.set_title('Height above terrain')    
    ax1.set_xlabel('Trajectory count')
    ax1.set_ylabel('Height (m)')
    plt.grid(linestyle='-')
    plt.tight_layout()
    plt.xlim(xmin=0, xmax=len(alt))
    plt.ylim(ymin=0, ymax=max(alt))
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot horizontal line
    plt.axhline(y=600., color='k', linewidth=2, alpha=0.5, label='Target height (600m.)' if count <= 2 else '_nolabel_')
    plt.axhline(y=50., color='m', linewidth=2, alpha=0.5, label='Minimum height (60m.)' if count <= 2 else '_nolabel_')

    # Plot distances
    ax1.plot(alt, c='r', label='Height (m)' if count <= 2 else '_nolabel_')
    leg = ax1.legend()

