import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import gym
import cogle_mavsim
from gym_recording.wrappers import TraceRecordingWrapper
import tensorflow as tf
from mpi4py import MPI
from datetime import datetime
import numpy as np
from baselines.qlearning import qlearning
from qlearning import build_state
#from functools import reduce

def run(env_id, seed, noise_type, layer_norm, evaluation, custom_log_dir, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    train_recording_path = os.path.join(custom_log_dir, env_id, 'train', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(train_recording_path)
    
    # Create envs.
    env = gym.make(env_id)
    env = TraceRecordingWrapper(env, directory=train_recording_path, buffer_batch_size=10)
    logger.info('TraceRecordingWrapper dir: {}'.format(env.directory))
    # env = bench.Monitor(env, os.path.join(train_recording_path, 'log'))

    if evaluation and rank==0:
        eval_recording_path = os.path.join(custom_log_dir, env_id, 'eval', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(eval_recording_path)
        
        eval_env = gym.make(env_id)
        eval_env = TraceRecordingWrapper(eval_env, directory=eval_recording_path, buffer_batch_size=10)
        logger.info('TraceRecordingWrapper eval dir: {}'.format(eval_env.directory))
        # eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        # env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('DDPG: rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='apl-nav-godiland-v0')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=(1e-4)/10.) #Optimially 100
    parser.add_argument('--critic-lr', type=float, default=(1e-3)/10.) # optimially 100
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=200)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=200)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=20)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--load-network-id', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=False)
    boolean_flag(parser, 'latest', default=False)
    boolean_flag(parser, 'plot-info', default=False)
    parser.add_argument('--custom-log-dir', type=str, default='./')
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


#if __name__ == '__main__':
    #args = parse_args()
    #if MPI.COMM_WORLD.Get_rank() == 0:
        #logger.configure()
    ## Run actual script.
    #run(**args)


#      QLEARNING MAIN
if __name__ == '__main__':
    args = parse_args()
    env = gym.make(args['env_id'])
    logger.set_level(logger.INFO)

    #env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
        # video_callable=lambda count: count % 10 == 0)

    #goal_average_steps = 255
    max_number_of_steps = 1000
    number_of_episodes = 1000
    last_time_steps_reward = np.ndarray(0)

    #n_bins = 8
    #n_bins_angle = 10

    number_of_features = env.observation_space.n
    #last_time_steps = np.ndarray(0)

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    #cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
    #pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
    #cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    #angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

    # The Q-learn algorithm
    qlearn = qlearning.QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.1)
    for i_episode in range(number_of_episodes):
        observation = env.reset()
        reward = 0
        state = build_state(observation)
        print("Episode: {}/{}".format(i_episode, number_of_episodes))

        #cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        #state = build_state([to_bin(cart_position, cart_position_bins),
                         #to_bin(pole_angle, pole_angle_bins),
                         #to_bin(cart_velocity, cart_velocity_bins),
                         #to_bin(angle_rate_of_change, angle_rate_bins)])

        for t in range(max_number_of_steps):
          if t > 1: # to have previous step reading
            if t % 10 == 0:
              print("step: {}/{}, Reward: {}".format(t, max_number_of_steps, reward))
            # env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            nextState = build_state(observation)
            #print(observation)
            #print(nextState)

            # Digitize the observation to get a state
            #cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
            #nextState = build_state([to_bin(cart_position, cart_position_bins),
                             #to_bin(pole_angle, pole_angle_bins),
                             #to_bin(cart_velocity, cart_velocity_bins),
                             #to_bin(angle_rate_of_change, angle_rate_bins)])

            # # If out of bounds
            # if (cart_position > 2.4 or cart_position < -2.4):
            #     reward = -200
            #     qlearn.learn(state, action, reward, nextState)
            #     print("Out of bounds, reseting")
            #     break
            
            if not(done) and t == max_number_of_steps - 1:
              done = True

            if not(done):
                qlearn.learn(state, action, reward, nextState)
                state = nextState
            else:
                # Q-learn stuff
                #reward = -1000 ## reward alway given by env, env when dead
                qlearn.learn(state, action, reward, nextState)
                last_time_steps_reward = np.append(last_time_steps_reward, [reward])
                break

    #l = last_time_steps_reward.tolist()
    print("Rewards of last_time_steps")
    print(last_time_steps_reward)
    #print(l.sort())
    #logger.info("Overall score: {:0.2f}".format(last_time_steps.mean()))
    #logger.info("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    #env.monitor.close()
    # gym.upload('/tmp/cartpole-experiment-1', algorithm_id='vmayoral simple Q-learning', api_key='your-key')
