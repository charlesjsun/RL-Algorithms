import gym
import numpy as np 
import sys
import time
import tensorflow as tf 

from vpg_continuous import ContinuousAgent
from vpg_discrete import DiscreteAgent

def reward_to_go(rewards, discount=0.99):
    n = len(rewards)
    rtg = np.zeros_like(rewards)
    for t in reversed(range(n)):
        rtg[t] = rewards[t] + discount * (rtg[t + 1] if t + 1 < n else 0)
    return rtg

def train_one_epoch(agent, env, batch_size, discount=0.99, max_ep_len=1000):
    ep_returns = []
    ep_lens = []

    states = []
    actions = []
    rewards_to_go = []

    curr_rewards = []
    s = env.reset()
    done = False

    ep_len = 0

    while True:
        a = agent.sample_action(s)
        new_s, r, done, _ = env.step(a)
        ep_len += 1

        states.append(s)
        actions.append(a)
        
        curr_rewards.append(r)
        
        s = new_s

        if done or ep_len == max_ep_len or len(states) >= batch_size:
            ep_returns.append(sum(curr_rewards))
            ep_lens.append(ep_len)

            if not done:
                curr_rewards[-1] = r + discount * agent.get_value(s)

            rewards_to_go += list(reward_to_go(curr_rewards, discount))
            
            if len(states) >= batch_size:
                break

            s = env.reset()
            curr_rewards = []
            done = False
            ep_len = 0

    states = np.array(states, dtype="float32")
    rewards_to_go = np.array(rewards_to_go, dtype="float32")
    actions = np.array(actions)

    agent.update(states, actions, rewards_to_go)

    return ep_returns, ep_lens
            
def train(agent, env, epochs, batch_size, save_path, save_freq=100, init_epoch=0, discount=0.99, max_ep_len=1000):
    for i in range(init_epoch, epochs):
        returns, lens = train_one_epoch(agent, env, batch_size, discount=discount, max_ep_len=max_ep_len)
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
    parser.add_argument("--v_lr", type=float, default=1e-3)
    parser.add_argument("--v_update_steps", type=int, default=80)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_layers", type=str, default="[64, 32]")
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    env = gym.make(args.env)

    hidden_layers = []
    if args.hidden_layers[1:-1].strip() != "":
        hidden_layers = [int(x) for x in args.hidden_layers[1:-1].split(",")]

    agent = None
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        agent = DiscreteAgent(env.observation_space.shape[0], env.action_space.n, hidden_layers=hidden_layers, 
                    policy_lr=args.pi_lr, v_lr=args.v_lr, v_update_steps=args.v_update_steps)
    else:
        agent = ContinuousAgent(env.observation_space.shape[0], env.action_space.shape[0], hidden_layers=hidden_layers, 
                    policy_lr=args.pi_lr, v_lr=args.v_lr, v_update_steps=args.v_update_steps)

    if args.load_epoch >= 0:
        agent.load(f"{args.path}_{args.load_epoch}")

    if args.epochs > 0 and not args.test_only:
        train(agent, env, args.epochs, args.bs, args.path, save_freq=args.save_freq, 
                init_epoch=args.init_epoch, discount=args.discount, max_ep_len=args.max_ep_len)

    if args.tests > 0:
        test_agent(agent, env, args.tests, 0.025)

    env.close()