'''
Q-learning implementation for APL Agent.
    @author: Simón C. Smith <artificialsimon@gmail.com>


Based on https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial4/
'''
import gym
import numpy
import random

class QLearn:
    '''A class for Q-Learning implementation.

        Based on [Barto and Sutton, 1998]

        :param list action: A list of ints coding the actions
        :param float epsilon: exploration constant
        :param float alpha: discount constant
        :param float gamma: discount factor
    '''
    def __init__(self, actions, epsilon, alpha, gamma, random_action):
        self.q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma 
        self.actions = actions
        self.random_action = random_action

    def getQ(self, state, action):
        """Returns the q-value for state action pair

            :param int state: A valid state
            :param int action: A valid action
            :returns: Q-value
            :rtype: float
        """
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        """Calculates new q-value

            Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))

            :param int state: The actual state
            :param int action: A valid action
            :param float reward: Reward obtained after performing action
            :param float value: Q(s,a)
        """
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = (1. - self.alpha) * self.q[(state, action)] + self.alpha * (value)

    def is_state_visited(self, state):
        """Returns whether a state exists in the q-value

            :param int state: The state to check
            :returns: If the state has been already added to the q-function
            :rtype: bool
        """
        flag = False
        for action in self.actions:
            if (state, action) in self.q:
                flag = True
                break
        return flag

    def distance_states(self, state_1, state_2):
        """Measures the distance between states

            Distance is weighted for different state
            features

            :param int state1: First state
            :param int state2: Second state
            :returns: Distance between first and second state
            :rtype: float
        """
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
        """Returns closest state to param state if it does not exists

            :param int state: The state to start the search
            :returns: The closest state
            :rtype: int
        """
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
            if temp_distance < distance:
                distance = temp_distance
                the_state = key_string
        return int(the_state)

    def chooseAction(self, state, return_q=False):
        """Returns action to be performed on state

            If the state has not been visited it will search
            for the closest one.

            :param int state: Actual state
            :param bool return_q: If return Q function
            :returns: action and Q function
            :rtype: int, list
        """
        q = [self.getQ(state, a) for a in self.actions]
        # TODO: closest_state only for exploitation mode, not learning
        if not self.is_state_visited(state):
            state = self.closest_state(state)
            q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)
        if random.random() < self.epsilon:
            i = random.choice(self.actions)
        else:
            count = q.count(maxQ)
            if count > 1:
                if self.random_action:
                    best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                    i = random.choice(best)
                else:
                    i = q.index(maxQ)
            else:
                i = q.index(maxQ)
        action = self.actions[i]
        if return_q:
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        """Updates the q-function

            :param int state1: State at actual time
            :param int action1: Action at actual time
            :param float reward: Reward after executing action1
            :param int state2: State at time t+1
        """
        _q = [self.getQ(state1, a) for a in self.actions]
        m_q2 = [self.getQ(state2, a) for a in self.actions]
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
        _q = [self.getQ(state1, a) for a in self.actions]


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]
