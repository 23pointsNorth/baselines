import cogle_mavsim
import gym
import math
import numpy as np


def calculate_action(obs):
    *ms, ra, dist, angle = obs
    print('MS: ' + str(ms))

    delta_angle = 180 / (len(ms) - 1)
    ms_angles = [i * delta_angle - 90 for i in range(len(ms))]

    if any(m > -0.9 for m in ms) and \
       -90 < math.degrees(angle * math.pi) and \
       math.degrees(angle * math.pi) < 90:
        dirs_error = [msa - math.degrees(angle * math.pi) for msa in ms_angles]
        angle = math.radians(dirs_error.index(min(dirs_error))) / math.pi

    alt_error = ra - 450 / 3000
    pitch = -max(min(20 * alt_error, 1), -1)
    yaw = -angle

    return np.array([pitch, yaw])


def expert_agent():
    env = gym.make('CoGLEM1-v0')
    obs = env.reset()

    done = False
    while not done:
            env.render()
            action = calculate_action(obs)
            print('Doing action: '), action, ' ', env._elapsed_steps, '\r'
            obs, reward, done, info = env.step(action)
            print('observations: ', obs, ' ', reward, ' ', done, '\r')

    env.close()


if __name__ == '__main__':
    expert_agent()
