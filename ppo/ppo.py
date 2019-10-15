import gym
import numpy as np 
import sys
import time
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

from utils import utils

class GAEBuffer:
    def __init__(self, batch_size, lam, discount, state_dim, action_dim):
        self.batch_size = batch_size
        self.lam = lam
        self.discount = discount
        
        self.states = np.zeros((batch_size, state_dim), dtype="float32")
        self.actions = np.zeros((batch_size, action_dim), dtype="float32")
        self.rewards = np.zeros(batch_size, dtype="float32")
        self.advs = np.zeros(batch_size, dtype="float32")
        self.log_probs = np.zeros(batch_size, dtype="float32")

        self.num = 0
        self.trajectory_start = 0

    def store(self, s, a, r):
        assert not self.is_full(), "Tried to store but buffer is full."
        self.states[self.num] = s
        self.actions[self.num] = a
        self.rewards[self.num] = r
        self.num += 1

    def is_full(self):
        return self.num == self.batch_size

    def calc_trajectory(self, agent, s, is_terminal):
        """ After an episode ends, calculate the GAE and rewards-to-go for this trajectory
        
        Args:
            agent: an instance of Agent with evaluate method
            s: the final state of the trajectory
            is_terminal: if s is not a terminal state (e.g. max_ep_len or batch_size reached), then use the 
                agent's value function to estimate the returns
        """
        trajectory = slice(self.trajectory_start, self.num)

        # calculate the state-values and log-probs of this trajectory
        values, self.log_probs[trajectory] = agent.evaluate(self.states[trajectory], self.actions[trajectory])
        final_val = 0 if is_terminal else agent.calc_state_value(s)
        values = np.append(values, final_val)

        # GAE calculations for actor update
        deltas = self.rewards[trajectory] + self.discount * values[1:] - values[:-1]
        self.advs[trajectory] = utils.discounted_cumsum(deltas, self.discount * self.lam)

        # rewards to go calculations for critic update, estimate final returns if not terminal
        self.rewards[self.num - 1] += self.discount * final_val
        self.rewards[trajectory] = utils.discounted_cumsum(self.rewards[trajectory], self.discount)

        # start new trajectory
        self.trajectory_start = self.num
    
    def batch_normalize_advs(self):
        assert self.is_full(), "Tried to batch normalize before buffer is full"
        advs_mean, advs_std = tf.reduce_mean(self.advs), tf.math.reduce_std(self.advs)
        self.advs = (self.advs - advs_mean) / advs_std

    def clear(self):
        self.num = 0
        self.trajectory_start = 0

class Agent:
    def __init__(self, input_dim, output_dim, hidden_layers=[32, 32], pi_lr=1e-3, pi_update_steps=80, v_lr=1e-3, v_update_steps=80):
        self.input_dim = input_dim
        self.output_dim = output_dim 

        self.pi_update_steps = pi_update_steps
        self.v_update_steps = v_update_steps

        self._build_policy_model(input_dim, output_dim, hidden_layers, pi_lr)
        self._build_value_model(input_dim, hidden_layers, v_lr)

    def _build_policy_model(self, input_dim, output_dim, hidden_layers, lr):
        policy_input = Input(shape=(input_dim,))
        X = policy_input
        for size in hidden_layers:
            X = Dense(size, activation="tanh", kernel_initializer="glorot_normal")(X)
        mu = Dense(output_dim, activation=None, kernel_initializer="zeros", use_bias=False)(X)
        self.policy = Model(inputs=policy_input, outputs=mu)
        self.log_stds = tf.Variable(-0.75 * np.ones((output_dim,)), dtype="float32", name="log_stds", trainable=True)
        self.policy_opt = Adam(lr)

    def _build_value_model(self, input_dim, hidden_layers, lr):
        v_input = Input(shape=(input_dim,))
        X = v_input
        for size in hidden_layers:
            X = Dense(size, activation="tanh", kernel_initializer="glorot_normal")(X)
        X = Dense(1, activation=None, kernel_initializer="glorot_normal")(X)
        self.v = Model(inputs=v_input, outputs=X)
        self.v_opt = Adam(lr)

    def update(self, policy_loss, v_loss):
        """Does one step of policy gradient update
        
        Args:
            states: np.array of sample states. dim = (n_samples, self.input_dim)
            action: np.array of sample actions. dim = (n_samples,)
            weights: np.array of sample weights e.g. rewards-to-go. dim = (n_samples,)
        """
        # Update the policy
        for _ in range(self.pi_update_steps):
            self.policy_opt.minimize(policy_loss, lambda: self.policy.trainable_weights + [self.log_stds])

        # Update the Value function
        for _ in range(self.v_update_steps):
            self.v_opt.minimize(v_loss, lambda: self.v.trainable_weights)

        stds = tf.exp(self.log_stds)
        print("Stds:", stds)

    def evaluate(self, states, actions):
        """ Returns the state-values and log probs of the given states and actions"""
        means = self.policy(states)
        log_probs = utils.gaussian_log_prob(actions, means, self.log_stds)
        values = self.v(states)
        return values, log_probs

    def sample_action(self, s):
        state = np.expand_dims(s, axis=0)
        means = self.policy.predict(state)[0]
        stds = tf.exp(self.log_stds)
        noises = tf.random.normal((self.output_dim,))
        sample = means + stds * noises
        return tf.clip_by_value(sample, -1, 1).numpy()

    def calc_state_value(self, s):
        state = np.expand_dims(s, axis=0)
        value = self.v.predict(state)[0]
        return value

    def save(self, path, extension="h5"):
        self.policy.save(f"{path}_pi.{extension}")
        self.v.save(f"{path}_v.{extension}")
        np.save(f"{path}_log_stds", self.log_stds.numpy())

    def load(self, path, extension="h5"):
        del self.policy
        self.policy = tf.keras.models.load_model(f"{path}_pi.{extension}")
        del self.v 
        self.v = tf.keras.models.load_model(f"{path}_v.{extension}")
        self.log_stds.assign(np.load(f"{path}_log_stds.npy"))

def train_one_epoch(agent, env, batch_size, lam=0.97, discount=0.99, max_ep_len=1000, eps_clip=0.2):
    ep_returns = []
    ep_lens = []
    curr_rewards = []

    s, r, done = env.reset(), 0, False

    ep_len = 0

    buffer = GAEBuffer(batch_size, lam, discount, agent.input_dim, agent.output_dim)

    while True:
        a = agent.sample_action(s)
        new_s, r, done, _ = env.step(a)
        
        buffer.store(s, a, r)
        
        s = new_s

        ep_len += 1
        curr_rewards.append(r)
    
        if done or ep_len == max_ep_len or buffer.is_full():
            ep_returns.append(sum(curr_rewards))
            ep_lens.append(ep_len)

            buffer.calc_trajectory(agent, s, done)

            if buffer.is_full():
                break

            s = env.reset()
            curr_rewards = []
            done = False
            ep_len = 0

    buffer.batch_normalize_advs()

    def policy_loss():
        means = agent.policy(buffer.states)
        log_probs = utils.gaussian_log_prob(buffer.actions, means, agent.log_stds)
        ratios = tf.exp(log_probs - buffer.log_probs)
        bounds = tf.where(buffer.advs >= 0, (1 + eps_clip) * buffer.advs, (1 - eps_clip) * buffer.advs)
        surrogate_losses = tf.minimum(ratios * buffer.advs, bounds)
        return -tf.reduce_mean(surrogate_losses)

    def v_loss():
        values = agent.v(buffer.states)
        return tf.reduce_mean(tf.math.squared_difference(values, buffer.rewards))

    agent.update(policy_loss, v_loss)
    buffer.clear()

    return ep_returns, ep_lens
            
def train(agent, env, epochs, batch_size, save_path, save_freq=100, init_epoch=0, lam=0.97, discount=0.99, max_ep_len=1000, eps_clip=0.2):
    for i in range(init_epoch, epochs):
        returns, lens = train_one_epoch(agent, env, batch_size, lam=lam, discount=discount, max_ep_len=max_ep_len)
        print(f"epoch: {i}, \t return: {np.mean(returns)}, \t episode length: {np.mean(lens)}")
        if i % save_freq == 0: 
            agent.save(f"{save_path}_{i}")
    agent.save(f"{save_path}_{epochs - 1}")

def test_agent(agent, env, n_tests, delay=1):
    for test in range(n_tests):
        print(f"Test #{test}")
        s = env.reset()
        done = False
        total_reward = 0
        while True:
            time.sleep(delay)
            env.render()
            a = agent.sample_action(s)
            print(f"Chose action {a} for state {s}")
            s, reward, done, _ = env.step(a)
            total_reward += reward
            if done:
                print(f"Done. Total Reward = {total_reward}")
                time.sleep(2)
                break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BipedalWalker-v2")
    parser.add_argument("--path", type=str, default="models/bipedalwalker_model")
    parser.add_argument("--load_epoch", type=int, default=-1)
    parser.add_argument("--tests", type=int, default=0)
    parser.add_argument("--init_epoch", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bs", type=int, default=2000)
    parser.add_argument("--pi_lr", type=float, default=1e-3)
    parser.add_argument("--pi_update_steps", type=int, default=80)
    parser.add_argument("--v_lr", type=float, default=1e-3)
    parser.add_argument("--v_update_steps", type=int, default=80)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lambda", type=float, default=0.97)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_layers", type=str, default="[64, 32]")
    parser.add_argument("--eps_clip", type=float, default=0.2)
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    env = gym.make(args.env)

    hidden_layers = []
    if args.hidden_layers[1:-1].strip() != "":
        hidden_layers = [int(x) for x in args.hidden_layers[1:-1].split(",")]

    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0], hidden_layers=hidden_layers, 
                    pi_lr=args.pi_lr, pi_update_steps=args.pi_update_steps, v_lr=args.v_lr, v_update_steps=args.v_update_steps)

    if args.load_epoch >= 0:
        agent.load(f"{args.path}_{args.load_epoch}")

    if args.epochs > 0 and not args.test_only:
        train(agent, env, args.epochs, args.bs, args.path, save_freq=args.save_freq, 
                init_epoch=args.init_epoch, discount=args.discount, max_ep_len=args.max_ep_len, eps_clip=args.eps_clip)

    if args.tests > 0:
        test_agent(agent, env, args.tests, 0.025)

    env.close()