# import gym
# import numpy as np
# import time

# """
# Qlearning is an off policy learning python implementation.
# This is a python implementation of the qlearning algorithm in the Sutton and
# Barto's book on RL. It's called SARSA because - (state, action, reward, state,
# action). The only difference between SARSA and Qlearning is that SARSA takes the
# next action based on the current policy while qlearning takes the action with
# maximum utility of next state.
# Using the simplest gym environment for brevity: https://gym.openai.com/envs/FrozenLake-v0/
# """

# def init_q(s, a, type="ones"):
#     """
#     @param s the number of states
#     @param a the number of actions
#     @param type random, ones or zeros for the initialization
#     """
#     if type == "ones":
#         return np.ones((s, a))
#     elif type == "random":
#         return np.random.random((s, a))
#     elif type == "zeros":
#         return np.zeros((s, a))


# def epsilon_greedy(Q, epsilon, n_actions, s, train=False):
#     """
#     @param Q Q values state x action -> value
#     @param epsilon for exploration
#     @param s number of states
#     @param train if true then no random actions selected
#     """
#     if train or np.random.rand() < epsilon:
#         action = np.argmax(Q[s, :])
#     else:
#         action = np.random.randint(0, n_actions)
#     return action

# def qlearning(alpha, gamma, epsilon, episodes, max_steps, n_tests, render = False, test=False):
#     """
#     @param alpha learning rate
#     @param gamma decay factor
#     @param epsilon for exploration
#     @param max_steps for max step in each episode
#     @param n_tests number of test episodes
#     """
#     env = gym.make('FrozenLake-v0')
#     n_states, n_actions = env.observation_space.n, env.action_space.n
#     Q = init_q(n_states, n_actions, type="ones")
#     timestep_reward = []
#     for episode in range(episodes):
#         print(f"Episode: {episode}")
#         s = env.reset()
#         a = epsilon_greedy(Q, epsilon, n_actions, s)
#         t = 0
#         total_reward = 0
#         done = False
#         while t < max_steps:
#             if render:
#                 env.render()
#             t += 1
#             s_, reward, done, info = env.step(a)
#             total_reward += reward
#             a_ = np.argmax(Q[s_, :])
#             if done:
#                 Q[s, a] += alpha * ( reward  - Q[s, a] )
#             else:
#                 Q[s, a] += alpha * ( reward + (gamma * Q[s_, a_]) - Q[s, a] )
#             s, a = s_, a_
#             if done:
#                 if render:
#                     print(f"This episode took {t} timesteps and reward: {total_reward}")
#                 timestep_reward.append(total_reward)
#                 break
#     if render:
#         print(f"Here are the Q values:\n{Q}\nTesting now:")
#     if test:
#         test_agent(Q, env, n_tests, n_actions)
#     return timestep_reward

# def test_agent(Q, env, n_tests, n_actions, delay=1):
#     for test in range(n_tests):
#         print(f"Test #{test}")
#         s = env.reset()
#         done = False
#         epsilon = 0
#         while True:
#             time.sleep(delay)
#             env.render()
#             a = epsilon_greedy(Q, epsilon, n_actions, s, train=True)
#             print(f"Chose action {a} for state {s}")
#             s, reward, done, info = env.step(a)
#             if done:
#                 if reward > 0:
#                     print("Reached goal!")
#                 else:
#                     print("Shit! dead x_x")
#                 time.sleep(3)
#                 break


# if __name__ =="__main__":
#     alpha = 0.4
#     gamma = 0.999
#     epsilon = 0.9
#     episodes = 10000
#     max_steps = 2500
#     n_tests = 2
#     timestep_reward = qlearning(alpha, gamma, epsilon, episodes, max_steps, n_tests, test = True)
#     print(timestep_reward)

# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
#       if builtins is not defined
# sudo pip install -U future

import gym
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import time

class StateDiscretizer:
    def __init__(self):
        # Note: to make this better you could look at how often each bin was
        # actually used while running the script.
        # It's not clear from the high/low values nor sample() what values
        # we really expect to get.
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9) # (-inf, inf) (I did not check that these were good values)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9) # (-inf, inf) (I did not check that these were good values)

    def discretize(self, obs):
        # returns an int
        cart_pos, cart_vel, pole_angle, pole_vel = obs
        return self.build_state([self.to_bin(cart_pos, self.cart_position_bins),
            self.to_bin(cart_vel, self.cart_velocity_bins),
            self.to_bin(pole_angle, self.pole_angle_bins),
            self.to_bin(pole_vel, self.pole_velocity_bins)])
            
    def to_bin(self, value, bins):
        return np.digitize(x=[value], bins=bins)[0]

    def build_state(self, features):
        return int("".join(map(lambda feature: str(int(feature)), features)))

class QTable:
    def __init__(self, n_states, n_actions):
        self.Q = np.random.uniform(low=-1, high=1, size=(n_states, n_actions))
        self.s2i = dict()
        self.i = 0

    def get(self, s, a):
        return self.Q[s, a]

    def set(self, s, a, q):
        self.Q[s, a] = q

    def get_Qs(self, s):
        return self.Q[s]

    def get_action(self, s):
        return np.argmax(self.get_Qs(s))

    def max_Q(self, s):
        return np.max(self.get_Qs(s))

class Model:
    def __init__(self, env):
        self.state_discretizer = StateDiscretizer()
        self.env = env
        self.q_table = QTable(10 ** env.observation_space.shape[0], env.action_space.n)

    def update(self, obs, new_obs, a, reward, alpha, gamma):
        s = self.state_discretizer.discretize(obs)
        new_s = self.state_discretizer.discretize(new_obs)

        G = reward + gamma * self.q_table.max_Q(new_s)
        q = self.q_table.get(s, a)

        self.q_table.set(s, a, q + alpha * (G - q))

    def sample_action(self, obs, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            s = self.state_discretizer.discretize(obs)
            return self.q_table.get_action(s)

    def save(self, path):
        np.savetxt(path, self.q_table.Q)

    def load(self, path):
        self.q_table.Q = np.loadtxt(path)


def play_episode(model, env, epsilon, alpha, gamma):
    obs = env.reset()
    done = False
    total_reward = 0
    i = 0
    while not done:
        a = model.sample_action(obs, epsilon)
        new_obs, reward, done, _ = env.step(a)

        total_reward += reward
        if done and i < 199:
            reward = -300

        model.update(obs, new_obs, a, reward, alpha, gamma)
        obs = new_obs

        i += 1

    return total_reward

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

def train_agent(model, env, n_train, do_save=True):
    N = n_train
    totalrewards = np.empty(N)
    alpha = 1e-2
    gamma = 0.9
    for n in range(N):
        epsilon = 0.5 / (1 + np.exp((n - n_train * 0.5) / (n_train * 0.2)))
        totalreward = play_episode(model, env, epsilon, alpha, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "epsilon:", epsilon)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    if (do_save):
        model.save('cartpole_model.txt')

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

def test_agent(model, env, n_tests, delay=1):
    for test in range(n_tests):
        print(f"Test #{test}")
        obs = env.reset()
        done = False
        epsilon = 0
        total_reward = 0
        while True:
            time.sleep(delay)
            env.render()
            a = model.sample_action(obs, epsilon)
            print(f"Chose action {a} for state {obs}")
            obs, reward, done, info = env.step(a)
            total_reward += reward
            if done:
                print(f"Done. Total Reward = {total_reward}")
                time.sleep(3)
                break

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    model = Model(env)

    if 'load' in sys.argv:
        model.load('cartpole_model.txt')
    else:
        train_agent(model, env, 30000)

    test_agent(model, env, 3, 0.025)

    env.close()
