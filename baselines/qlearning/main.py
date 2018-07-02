import argparse
import time
import os
import pickle
from baselines import logger
from baselines.common.misc_util import boolean_flag
from baselines.qlearning import qlearning

import gym
import cogle_mavsim
from gym_recording.wrappers import TraceRecordingWrapper
import tensorflow as tf
import numpy as np
from qlearning import build_state


def parse_args():
    """ Parse arguments from command line """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env-id', type=str, default='apl-nav-godiland-v0')
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--qfunction', type=str, default=None)
    boolean_flag(parser, 'learning', default=True)
    parser.add_argument('--epsilon', type=float, default=.3)
    boolean_flag(parser, 'drop_payload_agent', default=False)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if
    # we do specify them they agree with the other parameters
    if args.num_timesteps is not None:
        assert args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles *\
            args.nb_rollout_steps
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


def main():
    """ Q-learning """
    args = parse_args()
    env = gym.make(args['env_id'])
    logger.set_level(logger.INFO)
    max_number_of_steps = 10000
    number_of_episodes = 100000
    last_time_steps_reward = np.ndarray(0)
    # File to store que Q function
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = "./q_functions/" + timestr + '-' + args['env_id'] + ".qf"
    file_reward = "./rewards/" + timestr + ".csv"
    # Only dropping payload agent
    if args['drop_payload_agent']:
        logger.info('Exploitation drop payload agent selected')
        env.env.only_drop_payload_agent = True
        if args['qfunction'] is None:
            logger.error('Q-function not espicified')
            return 0
        # Turning off exploration
        args['epsilon'] = 0
        args['learning'] = False
        number_of_episodes = 1
    # The Q-learn algorithm
    qlearn = qlearning.QLearn(actions=range(env.action_space.n),
                              alpha=0.4, gamma=0.80, epsilon=args['epsilon'])
    # Loads Q function
    if args['qfunction'] is not None:
        try:
            with open(args['qfunction'], "rb") as file_p:   # Unpickling
                logger.info('Loading qfunction: %s' % args['qfunction'])
                qlearn.q = pickle.load(file_p)
                file_p.close()
        except IOError:
            logger.error('Q-Function file does not exists: %s'
                         % args['qfunction'])
            return 1
    episode_trace = []
    for i_episode in range(number_of_episodes):
        observation = env.reset()
        reward = 0
        state = build_state(observation)
        logger.info("Episode: %d/%d" % (i_episode + 1, number_of_episodes))
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
                                      info['self_state']['alt'],
                                      reward])
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
        trace_file.close()
        # Reward trace to file
        os.makedirs(os.path.dirname(file_reward), exist_ok=True)
        reward_file = open(file_reward, 'a')
        logger.info('Saving episode reward to: %s' % file_reward)
        reward_file.write("{}\n".format(reward))
        reward_file.close()


if __name__ == '__main__':
    main()
