import gym
import numpy as np 
import sys
import time
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

from utils import *
import scipy.signal

strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
# strategy = tf.distribute.MirroredStrategy()

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

    # def calc_trajectory(self, agent, s, is_terminal):
    #     """ After an episode ends, calculate the GAE and rewards-to-go for this trajectory
        
    #     Args:
    #         agent: an instance of Agent with evaluate method
    #         s: the final state of the trajectory
    #         is_terminal: if s is not a terminal state (e.g. max_ep_len or batch_size reached), then use the 
    #             agent's value function to estimate the returns
    #     """
    #     trajectory = slice(self.trajectory_start, self.num)

    #     # calculate the state-values and log-probs of this trajectory
    #     values, self.log_probs[trajectory] = agent.evaluate(self.states[trajectory], self.actions[trajectory])
    #     final_val = 0 if is_terminal else agent.calc_state_value(s)
    #     values = np.append(values, final_val)

    #     # GAE calculations for actor update
    #     deltas = self.rewards[trajectory] + self.discount * values[1:] - values[:-1]
    #     self.advs[trajectory] = discounted_cumsum(deltas, self.discount * self.lam)

    #     # rewards to go calculations for critic update, estimate final returns if not terminal
    #     self.rewards[self.num - 1] += self.discount * final_val
    #     self.rewards[trajectory] = discounted_cumsum(self.rewards[trajectory], self.discount)

    #     # start new trajectory
    #     self.trajectory_start = self.num

    def calc_trajectory(self, final_val):
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
        values = np.append(values, final_val)
        rewards = np.append(self.rewards[trajectory], final_val)

        # GAE calculations for actor update
        deltas = rewards[:-1] + self.discount * values[1:] - values[:-1]
        self.advs[trajectory] = discounted_cumsum(deltas, self.discount * self.lam)

        # rewards to go calculations for critic update, estimate final returns if not terminal
        self.rewards[trajectory] = discounted_cumsum(rewards, self.discount)[:-1]

        # start new trajectory
        self.trajectory_start = self.num
    
    def batch_normalize_advs(self, eps=1e-8):
        assert self.is_full(), "Tried to batch normalize before buffer is full"
        advs_mean, advs_std = np.mean(self.advs), np.std(self.advs)
        self.advs = (self.advs - advs_mean) / (advs_std + eps)

    def clear(self):
        self.num = 0
        self.trajectory_start = 0

class Agent:
    def __init__(self, input_dim, output_dim, hidden_layers=[32, 32], lr=1e-3, update_steps=80):
        self.input_dim = input_dim
        self.output_dim = output_dim 

        self.update_steps = update_steps
        self.opt = Adam(lr)

        with strategy.scope():
            self._build_policy_model(input_dim, output_dim, hidden_layers, lr)
            self._build_value_model(input_dim, hidden_layers, lr)

    def _build_policy_model(self, input_dim, output_dim, hidden_layers, lr):
        policy_input = Input(shape=(input_dim,))
        X = policy_input
        for size in hidden_layers:
            X = Dense(size, activation="tanh")(X)
        mu = Dense(output_dim, activation="tanh")(X)
        self.policy = Model(inputs=policy_input, outputs=mu)
        self.log_stds = tf.Variable(-0.75 * np.ones((output_dim,)), dtype="float32", name="log_stds", trainable=False)

    def _build_value_model(self, input_dim, hidden_layers, lr):
        v_input = Input(shape=(input_dim,))
        X = v_input
        for size in hidden_layers:
            X = Dense(size, activation="tanh")(X)
        X = Dense(1, activation=None)(X)
        self.vf = Model(inputs=v_input, outputs=X)

    def update(self, step_func):
        """Does one step of policy gradient update
        
        Args:
            states: np.array of sample states. dim = (n_samples, self.input_dim)
            action: np.array of sample actions. dim = (n_samples,)
            weights: np.array of sample weights e.g. rewards-to-go. dim = (n_samples,)
        """
        # Update the policy
        with strategy.scope():
            for _ in range(self.update_steps):
                step_func()

    def evaluate(self, states, actions):
        """ Returns the state-values and log probs of the given states and actions"""
        with strategy.scope():
            means = self.policy(states)
            log_probs = gaussian_log_prob(actions, means, self.log_stds)
            values = self.vf(states)
        return values, log_probs

    def evaluate_states(self, states):
        """ Returns the state-values of the given states"""
        with strategy.scope():
            values = self.vf(states)
        return values

    def evaluate_actions(self, states, actions):
        """ Returns the log probs of the actions given states"""
        with strategy.scope():
            means = self.policy(states)
            log_probs = gaussian_log_prob(actions, means, self.log_stds)
        return log_probs

    def sample_action(self, s):
        with strategy.scope():
            state = np.expand_dims(s, axis=0)
            means = self.policy.predict(state)[0]
            stds = tf.exp(self.log_stds)
            noises = tf.random.normal((self.output_dim,))
            sample = means + stds * noises
        return tf.clip_by_value(sample, -1, 1).numpy()

    def calc_state_value(self, s):
        with strategy.scope():
            state = np.expand_dims(s, axis=0)
            value = self.vf.predict(state)[0]
        return value

    def save(self, path, extension="h5"):
        self.policy.save(f"{path}_pi.{extension}")
        self.vf.save(f"{path}_v.{extension}")
        np.save(f"{path}_log_stds", self.log_stds.numpy())

    def load(self, path, extension="h5"):
        with strategy.scope():
            del self.policy
            self.policy = tf.keras.models.load_model(f"{path}_pi.{extension}")
            del self.vf 
            self.vf = tf.keras.models.load_model(f"{path}_v.{extension}")
            self.log_stds.assign(np.load(f"{path}_log_stds.npy"))

def train_one_epoch(agent, env, buffer, max_ep_len=1000, eps_clip=0.2, vf_coef=0.5):
    ep_returns = []
    ep_lens = []
    curr_rewards = []

    s, r, done = env.reset(), 0, False

    ep_len = 0

    while True:
        # a = agent.sample_action(s)
        # new_s, r, done, _ = env.step(a)
        # buffer.store(s, a, r)
        # s = new_s

        a = agent.sample_action(s)
        buffer.store(s, a, r)
        s, r, done, _ = env.step(a)

        ep_len += 1
        curr_rewards.append(r)
    
        if done or ep_len == max_ep_len or buffer.is_full():
            if done or ep_len == max_ep_len:
                ep_returns.append(sum(curr_rewards))
                ep_lens.append(ep_len)

            # buffer.calc_trajectory(agent, s, done)

            final_val = r if done else agent.calc_state_value(s)
            buffer.calc_trajectory(final_val)

            if buffer.is_full():
                break

            s = env.reset()
            curr_rewards = []
            done = False
            ep_len = 0
            r = 0

    buffer.batch_normalize_advs()
    
    @tf.function
    def step_func():
        def step():
            def loss():
                means = agent.policy(buffer.states)
                log_probs = gaussian_log_prob(buffer.actions, means, agent.log_stds)
                ratios = tf.exp(log_probs - buffer.log_probs)
                bounds = tf.where(buffer.advs > 0, (1 + eps_clip) * buffer.advs, (1 - eps_clip) * buffer.advs)
                surrogate_losses = tf.minimum(ratios * buffer.advs, bounds)
                policy_loss = -tf.reduce_mean(surrogate_losses)

                values = agent.vf(buffer.states)
                vf_loss = tf.reduce_mean(tf.math.squared_difference(values, buffer.rewards))

                return policy_loss + vf_coef * vf_loss
            agent.opt.minimize(loss, lambda: agent.policy.trainable_weights + agent.vf.trainable_weights)
        strategy.experimental_run_v2(step)

    agent.update(step_func)
    buffer.clear()

    return ep_returns, ep_lens

def train(agent=None, env=None, episodes=10000, batch_size=4000, save_path=None, save_freq=100, log_freq=5, init_ep=0, 
        lam=0.97, discount=0.99, max_ep_len=1000, eps_clip=0.2, vf_coef=0.5):

    buffer = GAEBuffer(batch_size, lam, discount, agent.input_dim, agent.output_dim)
    for i in range(init_ep, episodes):
        start = time.time()
        returns, lens = train_one_epoch(agent, env, buffer, max_ep_len=max_ep_len, eps_clip=eps_clip, vf_coef=vf_coef)
        end = time.time()

        ep_returns = []
        ep_lens = []
        curr_rewards = []

        s, r, done = env.reset(), 0, False

        ep_len = 0

        while True:
            # a = agent.sample_action(s)
            # new_s, r, done, _ = env.step(a)
            # buffer.store(s, a, r)
            # s = new_s

            a = agent.sample_action(s)
            buffer.store(s, a, r)
            s, r, done, _ = env.step(a)

            ep_len += 1
            curr_rewards.append(r)
        
            if done or ep_len == max_ep_len or buffer.is_full():
                if done or ep_len == max_ep_len:
                    ep_returns.append(sum(curr_rewards))
                    ep_lens.append(ep_len)

                # buffer.calc_trajectory(agent, s, done)

                final_val = r if done else agent.calc_state_value(s)
                buffer.calc_trajectory(final_val)

                if buffer.is_full():
                    break

                s = env.reset()
                curr_rewards = []
                done = False
                ep_len = 0
                r = 0

        buffer.batch_normalize_advs()
        
        @tf.function
        def step_func():
            def step():
                def loss():
                    means = agent.policy(buffer.states)
                    log_probs = gaussian_log_prob(buffer.actions, means, agent.log_stds)
                    ratios = tf.exp(log_probs - buffer.log_probs)
                    bounds = tf.where(buffer.advs > 0, (1 + eps_clip) * buffer.advs, (1 - eps_clip) * buffer.advs)
                    surrogate_losses = tf.minimum(ratios * buffer.advs, bounds)
                    policy_loss = -tf.reduce_mean(surrogate_losses)

                    values = agent.vf(buffer.states)
                    vf_loss = tf.reduce_mean(tf.math.squared_difference(values, buffer.rewards))

                    return policy_loss + vf_coef * vf_loss
                agent.opt.minimize(loss, lambda: agent.policy.trainable_weights + agent.vf.trainable_weights)
            strategy.experimental_run_v2(step)

        agent.update(step_func)
        buffer.clear()

        print(f"{end - start}s, \t epoch: {i}, \t return: {np.mean(returns)}, \t episode length: {np.mean(lens)} \t {np.exp(agent.log_stds.numpy())}")
        if i % save_freq == 0: 
            agent.save(f"{save_path}_{i}")
        if np.mean(returns) >= 300:
            break
    agent.save(f"{save_path}_{i}")

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

#python ppo.py --save_freq 50 --epochs 1500 --bs 4000 --max_ep_len 1500 --discount 0.99 --lam 0.97 --eps_clip 0.2 --seed 123 --pi_lr 3e-4 --pi_update_steps 80 --v_lr 1e-3 --v_update_steps 80 --hidden_layers "[64, 32]"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BipedalWalker-v2")
    parser.add_argument("--path", type=str, default="models/bipedalwalker_model")
    parser.add_argument("--load_ep", type=int, default=-1)
    parser.add_argument("--tests", type=int, default=0)
    parser.add_argument("--init_epoch", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--log_freq", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--bs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--update_steps", type=int, default=80)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_layers", type=str, default="[64, 32]")
    parser.add_argument("--eps_clip", type=float, default=0.2)
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    env = gym.make(args.env)
    env.seed(args.seed)

    hidden_layers = []
    if args.hidden_layers[1:-1].strip() != "":
        hidden_layers = [int(x) for x in args.hidden_layers[1:-1].split(",")]

    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0], hidden_layers=hidden_layers, 
                    lr=args.lr, update_steps=args.update_steps)

    if args.load_epoch >= 0:
        agent.load(f"{args.path}_{args.load_epoch}")

    if args.epochs > 0 and not args.test_only:
        train(agent=agent, env=env, episodes=args.episode, batch_size=args.bs, save_path=args.path, save_freq=args.save_freq, 
            log_freq=args.log_freq, discount=args.discount, lam=args.lam, init_ep=args.init_ep, max_ep_len=args.max_ep_len, 
            eps_clip=args.eps_clip, vf_coef=args.vf_coef)

    if args.tests > 0:
        test_agent(agent, env, args.tests, 0.025)

    env.close()