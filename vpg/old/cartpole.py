import gym
import numpy as np 
import sys
import time
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

class Agent:
    def __init__(self, input_dim, output_dim, hidden_layers=[32, 32], policy_lr=1e-3, v_lr=1e-3, v_update_steps=80):
        self.input_dim = input_dim
        self.output_dim = output_dim 

        self.v_update_steps = v_update_steps

        self._build_policy_model(input_dim, output_dim, hidden_layers, policy_lr)
        self._build_value_model(input_dim, hidden_layers, v_lr)

    def _build_policy_model(self, input_dim, output_dim, hidden_layers, lr):
        policy_input = Input(shape=(input_dim,))
        X = policy_input
        for size in hidden_layers:
            X = Dense(size, activation="relu")(X)
        X = Dense(output_dim, activation="softmax")(X)
        self.policy = Model(inputs=policy_input, outputs=X)
        self.policy_opt = Adam(lr)

    def _build_value_model(self, input_dim, hidden_layers, lr):
        v_input = Input(shape=(input_dim,))
        X = v_input
        for size in hidden_layers:
            X = Dense(size, activation="tanh")(X)
        X = Dense(1)(X)
        self.v = Model(inputs=v_input, outputs=X)
        self.v_opt = Adam(lr)

    def update(self, states, actions, rewards_to_go):
        """Does one step of policy gradient update
        
        Args:
            states: np.array of sample states. dim = (n_samples, self.input_dim)
            action: np.array of sample actions. dim = (n_samples,)
            weights: np.array of sample weights e.g. rewards-to-go. dim = (n_samples,)
        """
        # Update the policy
        def policy_loss():
            action_prob = self.policy(states)
            action_mask = utils.to_categorical(actions, num_classes=self.output_dim)
            probs = tf.reduce_sum(action_prob * action_mask, axis=1)
            log_probs = tf.math.log(probs)
            advs = rewards_to_go - self.v(states)
            return -tf.reduce_mean(log_probs * advs)

        self.policy_opt.minimize(policy_loss, lambda: self.policy.trainable_weights)

        # Update the Value function
        def v_loss():
            values = self.v(states)
            return tf.reduce_mean(tf.math.squared_difference(values, rewards_to_go))

        for _ in range(self.v_update_steps):
            self.v_opt.minimize(v_loss, lambda: self.v.trainable_weights)

    def sample_action(self, s):
        """"""
        state = np.expand_dims(s, axis=0)
        action_prob = self.policy.predict(state)[0]
        return np.random.choice(range(self.output_dim), p=action_prob)

    def save(self, path, extension="h5"):
        self.policy.save(f"{path}_pi.{extension}")
        self.v.save(f"{path}_v.{extension}")

    def load(self, path, extension="h5"):
        del self.policy
        self.policy = tf.keras.models.load_model(f"{path}_pi.{extension}")
        del self.v 
        self.v = tf.keras.models.load_model(f"{path}_v.{extension}")

def reward_to_go(rewards, discount=0.99):
    n = len(rewards)
    rtg = np.zeros_like(rewards)
    for t in reversed(range(n)):
        rtg[t] = rewards[t] + discount * (rtg[t + 1] if t + 1 < n else 0)
    return rtg

def train_one_epoch(agent, env, batch_size, discount=0.99):
    ep_returns = []
    ep_lens = []

    states = []
    actions = []
    rewards_to_go = []

    curr_rewards = []
    s = env.reset()
    done = False

    while True:
        a = agent.sample_action(s)
        new_s, r, done, _ = env.step(a)

        states.append(s)
        actions.append(a)
        
        curr_rewards.append(r)

        s = new_s

        if done:
            ep_return = sum(curr_rewards)
            ep_len = len(curr_rewards)

            ep_returns.append(ep_return)
            ep_lens.append(ep_len)

            rewards_to_go += list(reward_to_go(curr_rewards, discount))
            
            if len(states) >= batch_size:
                break

            s = env.reset()
            curr_rewards = []
            done = False

    states = np.array(states, dtype="float32")
    rewards_to_go = np.array(rewards_to_go, dtype="float32")
    actions = np.array(actions)

    agent.update(states, actions, rewards_to_go)

    return ep_returns, ep_lens
            
def train(agent, env, epochs, batch_size, save_path, save_freq=100, init_epoch=0, discount=0.99):
    for i in range(init_epoch, epochs):
        returns, lens = train_one_epoch(agent, env, batch_size, discount)
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
    parser.add_argument("--path", type=str, default="models/cartpole_model")
    parser.add_argument("--load_epoch", type=int, default=-1)
    parser.add_argument("--tests", type=int, default=3)
    parser.add_argument("--init_epoch", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--bs", type=int, default=200)
    parser.add_argument("--pi_lr", type=float, default=1e-3)
    parser.add_argument("--v_lr", type=float, default=1e-3)
    parser.add_argument("--v_update_steps", type=int, default=80)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--discount", type=float, default=0.99)
    args = parser.parse_args()

    env = gym.make('CartPole-v0')
    agent = Agent(env.observation_space.shape[0], env.action_space.n, hidden_layers=[16, 16], 
                    policy_lr=args.pi_lr, v_lr=args.v_lr, v_update_steps=args.v_update_steps)

    if args.load_epoch >= 0:
        agent.load(f"{args.path}_{args.load_epoch}")
    
    if args.epochs > 0 and not args.test_only:
        train(agent, env, args.epochs, args.bs, args.path, save_freq=args.save_freq, 
                init_epoch=args.init_epoch, discount=args.discount)

    if args.tests > 0:
        test_agent(agent, env, args.tests, 0.025)

    env.close()
