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
    def __init__(self, input_dim, output_dim, hidden_layers=[32, 32], lr=1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim 

        # Build the policy network
        self.input = Input(shape=(input_dim,))
        X = self.input
        for size in hidden_layers:
            X = Dense(size, activation="relu")(X)
        X = Dense(output_dim, activation="softmax")(X)
        self.model = Model(inputs=self.input, outputs=X)

        # Build the optimizer
        self.optimizer = Adam(lr)

    def update(self, states, actions, weights):
        """Does one step of policy gradient update
        
        Args:
            states: np.array of sample states. dim = (n_samples, self.input_dim)
            action: np.array of sample actions. dim = (n_samples,)
            weights: np.array of sample weights e.g. rewards-to-go. dim = (n_samples,)
        """
        def loss():
            action_prob = self.model(states)
            action_mask = utils.to_categorical(actions, num_classes=self.output_dim)
            probs = tf.reduce_sum(action_prob * action_mask, axis=1)
            log_probs = tf.math.log(probs)
            return -tf.reduce_mean(log_probs * weights)
        
        self.optimizer.minimize(loss, lambda: self.model.trainable_weights)

    def sample_action(self, s):
        """"""
        state = np.expand_dims(s, axis=0)
        action_prob = self.model.predict(state)[0]
        return np.random.choice(range(self.output_dim), p=action_prob)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        del self.model
        self.model = tf.keras.models.load_model(path)

def reward_to_go(rewards):
    n = len(rewards)
    rtgs = np.zeros_like(rewards)
    for i in reversed(range(n)):
        rtgs[i] = rewards[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def train_one_epoch(agent, env, batch_size):
    ep_returns = []
    ep_lens = []

    states = []
    actions = []
    weights = []

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

            weights += list(reward_to_go(curr_rewards))
            
            if len(states) >= batch_size:
                break

            s = env.reset()
            curr_rewards = []
            done = False

    states = np.array(states, dtype="float32")
    weights = np.array(weights, dtype="float32")
    actions = np.array(actions)

    agent.update(states, actions, weights)

    return ep_returns, ep_lens
            
def train(agent, env, epochs, batch_size, save_path, save_period=100, init_epoch=0):
    for i in range(init_epoch, epochs):
        returns, lens = train_one_epoch(agent, env, batch_size)
        print(f"epoch: {i}, \t return: {np.mean(returns)}, \t episode length: {np.mean(lens)}")
        if i % save_period == 0: 
            agent.save(f"{save_path}_{i}.h5")
    agent.save(f"{save_path}_{epochs - 1}.h5")

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
    parser.add_argument("--path", type=str, default="basic_models/cartpole_basic_model")
    parser.add_argument("--load_epoch", type=int, default=-1)
    parser.add_argument("--tests", type=int, default=10)
    parser.add_argument("--init_epoch", type=int, default=0)
    parser.add_argument("--save_period", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--bs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    env = gym.make('CartPole-v0')
    agent = Agent(env.observation_space.shape[0], env.action_space.n, [16, 16], args.lr)

    if args.load_epoch >= 0:
        agent.load(f"{args.path}_{args.load_epoch}.h5")
    
    if args.epochs > 0 and not args.test_only:
        train(agent, env, args.epochs, args.bs, args.path, save_period=args.save_period, init_epoch=args.init_epoch)

    if args.tests > 0:
        test_agent(agent, env, args.tests, 0.025)

    env.close()
