'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning

Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA

        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import gym
import numpy
import random

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma, random_action):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        self.random_action = random_action # on equal Q-value choose random

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
            #print("New state. Copying reward")
        else:
            #self.q[(state, action)] = oldv + self.alpha * (value - oldv)
            self.q[(state, action)] = (1. - self.alpha) * self.q[(state, action)] + self.alpha * (value)

    def is_state_visited(self, state):
        flag = False
        for action in self.actions:
            if (state, action) in self.q:
                flag = True
                break
        return flag

    def distance_states(self, state_1, state_2):
        weights = [1., 1., 1., .2, 1., .2]
        distance = .0
        for index, weight in enumerate(weights):
            temp_dist = abs(int(state_1[index]) - int(state_2[index]))
            if index == 3 or index == 5:  # headings are circular
                temp_dist = 8 - temp_dist
            if index == 1:  # normalisin hazard
                temp_dist /= 2
            if index == 2:  # normalisin distance to hiker
                temp_dist /= 3
            distance += weight * temp_dist
        return distance

    def closest_state(self, state):
        distance = 100.
        the_state = 0
        state_key = []
        state_string = str(state)
        while len(state_string) < 6:
            state_string = '0' + state_string
        for key in self.q:
            key_string = str(key[0])
            while len(key_string) < 6:
                key_string = '0' + key_string
            temp_distance = self.distance_states(state_string,
                                                 key_string)
            #print('state: ', key_string, ', distance:', temp_distance)
            if temp_distance < distance:
                distance = temp_distance
                the_state = key_string
        #print('best distance: ', distance, ', state: ', the_state)
        return int(the_state)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        #print("state: ", state)
        #if not self.is_state_visited(state):
            ##print("state not q found: ", state)
            #state = self.closest_state(state)
            ##print("closests state: ", state)
            #q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)
        if random.random() < self.epsilon:
            #print("choosing randomly")
            #print(self.actions)
            i = random.choice(self.actions)
            #print("random action: {}".format(i))
            #minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            ## add random values to all the actions, recalculate maxQ
            #q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            #maxQ = max(q)
        else:
          count = q.count(maxQ)
          # In case there're several state-action max values
          # we select a random one among them
          if count > 1:
            if self.random_action:
              best = [i for i in range(len(self.actions)) if q[i] == maxQ]
              #print("More than one action to choose from: Choosing randmly")
              #print(best)
              i = random.choice(best)
            else:
              #print("More than one action to choose from: Choosing first")
              i = q.index(maxQ)
          else:
              #print("Only one max Q-value to choose from")
              i = q.index(maxQ)
        action = self.actions[i]
        #print("QLearning Action: {}, state: {}".format(action, state))
        print(q)
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):

        #print("QLearning Learning Action: {}, state1: {}, reward: {}, state2: {}".format(action1, state1, reward, state2))
        _q = [self.getQ(state1, a) for a in self.actions]
        #print("q function on state1 prior leaerning")
        #print(_q)
        m_q2 = [self.getQ(state2, a) for a in self.actions]
        #print("q function on next state")
        #print(m_q2)
       
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
       
        #print("q function on state1 after leaerning")
        _q = [self.getQ(state1, a) for a in self.actions]
        #print(_q)
        #print("\n")

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

