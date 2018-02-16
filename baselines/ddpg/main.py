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

def run(env_id, seed, noise_type, layer_norm, evaluation, custom_log_dir, duo, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    recording_path = os.path.join(custom_log_dir, env_id, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(recording_path)
    
    # Create envs.
    env = gym.make(env_id)
    env = TraceRecordingWrapper(env, directory=recording_path, buffer_batch_size=10)
    print('TraceRecordingWrapper dir: {}'.format(env.directory))
    env = bench.Monitor(env, os.path.join(recording_path, 'log'))

    if evaluation and rank==0:
        eval_env = gym.make(env_id)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
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

    # Salient info
    from gym import spaces
    salient_action_space = spaces.Box(low=np.array([0.] * env.observation_space.shape[0]), 
                                       high=np.array([1.] * env.observation_space.shape[0]))
    print(salient_action_space, salient_action_space.shape[-1])
    salient_memory = Memory(limit=int(1e6), action_shape=salient_action_space.shape, observation_shape=env.observation_space.shape)
    salient_critic = Critic(layer_norm=layer_norm, name='salient_critic')
    salient_actor = Actor(salient_action_space.shape[-1], layer_norm=layer_norm, name='salient_actor')

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
    if duo:
        training.train_duo(env=env, eval_env=eval_env, param_noise=param_noise, salient_action_space=salient_action_space,
            action_noise=action_noise, actor=actor, critic=critic, memory=memory, salient_actor=salient_actor, salient_critic=salient_critic, salient_memory=salient_memory, **kwargs)
    else:
        training.train(env=env, eval_env=eval_env, param_noise=param_noise,
                       action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#    parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
    parser.add_argument('--env-id', type=str, default='CoGLEM1-virtual-v0')
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
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=60)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=00)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=20)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--load-network-id', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=False)
    boolean_flag(parser, 'latest', default=False)
    boolean_flag(parser, 'duo', default=False)
    parser.add_argument('--custom-log-dir', type=str, default='./')
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    run(**args)
