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
import pickle


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
    parser.add_argument('--qfunction', type=str, default = None)
    boolean_flag(parser, 'learning', default=True)
    parser.add_argument('--epsilon', type=float, default = .3)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


def main():
    """ Q-learning """
    args = parse_args()
    env = gym.make(args['env_id'])
    logger.set_level(logger.INFO)
    max_number_of_steps = 10000
    number_of_episodes = 1000
    last_time_steps_reward = np.ndarray(0)
    # number_of_features = env.observation_space.n
    # File to store que Q function
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = "./q_functions/" + timestr + ".qf"
    # The Q-learn algorithm
    qlearn = qlearning.QLearn(actions=range(env.action_space.n),
                              alpha=0.4, gamma=0.80, epsilon=args['epsilon'])
    # Loads Q function
    if args['qfunction'] is not None:
        with open(args['qfunction'], "rb") as file_p:   # Unpickling
            logger.info('Loading qfunction: %s' % args['qfunction'])
            qlearn.q = pickle.load(file_p)
            file_p.close()

    episode_trace = []
    for i_episode in range(number_of_episodes):
        observation = env.reset()
        reward = 0
        state = build_state(observation)
        logger.info("Episode: %d/%d" % (i_episode, number_of_episodes))
        if args['learning']:
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "wb") as file_p:   # Pickling
                logger.info('Saving Q function to file: %s' % file_name)
                pickle.dump(qlearn.q, file_p)
                file_p.close()
        for step_t in range(max_number_of_steps):
            if step_t > 1:  # to have previous step reading
                if step_t % 10 == 0:
                    logger.info("step: %d/%d" % (step_t, max_number_of_steps))
                # Pick an action based on the current state
                action = qlearn.chooseAction(state)
                # Execute the action and get feedback
                observation, reward, done, info = env.step(action)
                next_state = build_state(observation)
                episode_trace.append([info['self_state']['lon'],
                                      info['self_state']['lat'],
                                      info['self_state']['alt']])
                # print(observation)
                # print(next_state)
                if not(done) and step_t == max_number_of_steps - 1:
                    done = True
                if not done:
                    if args['learning']:
                        qlearn.learn(state, action, reward, next_state)
                    state = next_state
                else:
                    # Q-learn stuff
                    if args['learning']:
                        qlearn.learn(state, action, reward, next_state)
                    last_time_steps_reward = np.append(last_time_steps_reward,
                                                       [reward])
                    step_t = max_number_of_steps - 1
                if done:
                    break  # TODO: get rid of all breaks
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_trace = "./traces/" + timestr + ".csv"
        os.makedirs(os.path.dirname(file_trace), exist_ok=True)
        trace_file = open(file_trace, 'w')
        logger.info('Saving trace of episode in: %s' % file_trace)
        for item in episode_trace:
            trace_file.write("{}, {}, {}\n".format(item[0], item[1], item[2]))
        del episode_trace[:]


if __name__ == '__main__':
    main()
