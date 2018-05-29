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
    max_number_of_steps = 10000
    number_of_episodes = 1000
    last_time_steps_reward = np.ndarray(0)
    number_of_features = env.observation_space.n
    # File to store que Q function
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fileName = "./q_functions/" + timestr + ".qf"
    # The Q-learn algorithm
    qlearn = qlearning.QLearn(actions=range(env.action_space.n),
                    alpha=0.4, gamma=0.80, epsilon = args['epsilon'])
    #Loads Q function
    if args['qfunction'] != None:
      with open(args['qfunction'], "rb") as fp:   # Unpickling
        qlearn.q = pickle.load(fp)
        fp.close()

    episode_trace = []

    for i_episode in range(number_of_episodes):
        observation = env.reset()
        reward = 0
        state = build_state(observation)
        print("Episode: {}/{}".format(i_episode, number_of_episodes))

        if args['learning']:
          os.makedirs(os.path.dirname(fileName), exist_ok=True)
          with open(fileName, "wb") as fp:   #Pickling
            logger.info('Saving Q function to file: {}'.format(fileName))
            pickle.dump(qlearn.q, fp)
            fp.close()

        for t in range(max_number_of_steps):
          if t > 1: # to have previous step reading
            if t % 10 == 0:
              #print("step: {}/{}, Local Reward: {}".format(t, max_number_of_steps, reward))
              print("step: {}/{}".format(t, max_number_of_steps))
            # env.render()
            #print(qlearn.q)

            # Pick an action based on the current state
            #print(state)
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            nextState = build_state(observation)
            #print(info)
            episode_trace.append([info['self_state']['lon'], info['self_state']['lat'], info['self_state']['alt']])

            #print(observation)
            #print(nextState)
            
            if not(done) and t == max_number_of_steps - 1:
              done = True

            if not(done):
                if args['learning']:
                  qlearn.learn(state, action, reward, nextState)
                state = nextState
            else:
                # Q-learn stuff
                #reward = -1000 ## reward alway given by env, env when dead
                if args['learning']:
                  qlearn.learn(state, action, reward, nextState)
                last_time_steps_reward = np.append(last_time_steps_reward, [reward])
                t = max_number_of_steps - 1
            # Change of Context
            if info['change_of_context']:
              qlearn.actions = range(info['new_context_n_actions'])
              state = build_state(info['obs_context_change'])

            if done:
              break # TODO: get rid of all breaks

        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_trace = "./traces/" + timestr + ".csv"
        os.makedirs(os.path.dirname(file_trace), exist_ok=True)
        trace_file = open(file_trace, 'w')
        logger.info('Saving trace of episode in: {}'.format(file_trace))
        for item in episode_trace:
            trace_file.write("{}, {}, {}\n".format(item[0], item[1], item[2]))
        del episode_trace[:]

    #l = last_time_steps_reward.tolist()
    #print("Rewards of last_time_steps")
    #print(last_time_steps_reward)
    #print(l.sort())
    #logger.info("Overall score: {:0.2f}".format(last_time_steps.mean()))
    #logger.info("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    #env.monitor.close()
    # gym.upload('/tmp/cartpole-experiment-1', algorithm_id='vmayoral simple Q-learning', api_key='your-key')
