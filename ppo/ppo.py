import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np 
import gym
import pybullet_envs
import time

from core.buffers import GAEBuffer

from core.agents import Agent
from core.agents import GaussianPolicy
from core.agents import ValueFunction

from core.utils import test_agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent(Agent):
    def __init__(self, state_dim, action_dim, init_std=0.5, hidden_layers=[32, 32]):
        super(PPOAgent, self).__init__(state_dim, action_dim)
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_layers, init_std)
        self.vf = ValueFunction(state_dim, hidden_layers)

    def evaluate(self, states, actions):
        """ Returns the state-values and log probs of the given states and actions

        Args:
            states: (n, state_dim) tensor
            actions: (n, action_dim) tensor

        Returns:
            (n,) tensor of state values,
            (n,) tensor of log probs
        """
        means, covs = self.policy(states)

        dist = MultivariateNormal(means, covs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        values = torch.squeeze(self.vf(states))

        return values, log_probs, entropy

    def evaluate_states(self, states):
        """ Returns the state-values of the given states
        
        Args:
            states: (n, state_dim) tensor

        Returns:
            (n,) tensor of state values,
        """
        values = torch.squeeze(self.vf(states))
        return values

    def sample_action(self, state):
        """ Returns an action based on the current policy given state

        Args:
            state: (state_dim,) tensor

        Returns:
            (action_dim,) tensor
        """
        mean, cov = self.policy(state)
        dist = MultivariateNormal(mean, cov)
        return dist.sample()

    def sample_action_numpy(self, state):
        """ Returns an action based on the current policy given state

        Args:
            state: (state_dim,) numpy array

        Returns:
            (action_dim,) numpy array
        """
        state = torch.Tensor(state).to(device)
        action = self.sample_action(state)
        return action.cpu().data.numpy()

def train(agent=None, env=None, episodes=10000, batch_size=4000, save_path=None, save_freq=100, init_ep=0, 
        lam=0.97, discount=0.99, max_ep_len=1000, eps_clip=0.2, vf_coef=0.5, lr=3e-4, update_steps=80):

    buffer = GAEBuffer(batch_size, lam, discount, agent.state_dim, agent.action_dim, agent, device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    def update():
        old_states, old_actions, old_rewards, old_advs, old_log_probs = buffer.get_buffer()

        for _ in range(update_steps):
            values, log_probs, entropy = agent.evaluate(old_states, old_actions)
            
            # calculate policy loss
            ratios = torch.exp(log_probs - old_log_probs)
            bounds = torch.where(old_advs > 0, (1 + eps_clip) * old_advs, (1 - eps_clip) * old_advs)
            surrogate_losses = torch.min(ratios * old_advs, bounds)
            # policy_loss = -torch.mean(surrogate_losses)

            # calculate value function loss
            vf_squared_error = (values - old_rewards) ** 2

            # total loss
            loss = torch.mean(-surrogate_losses + vf_coef * vf_squared_error - 0.01 * entropy)

            # optimize one step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ep = init_ep
    last_save = ep
    
    while ep < episodes:
        start = time.time()

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

            a = agent.sample_action_numpy(s)
            buffer.store_numpy(s, a, r)
            s, r, done, _ = env.step(a)

            ep_len += 1
            curr_rewards.append(r)
        
            if done or ep_len == max_ep_len or buffer.is_full():
                if done or ep_len == max_ep_len:
                    ep_returns.append(sum(curr_rewards))
                    ep_lens.append(ep_len)

                # buffer.calc_trajectory(agent, s, done)

                final_val = r if done else agent.evaluate_states(torch.tensor(s).to(device)).item()
                buffer.calc_trajectory(final_val)

                ep += 1
                ep_len = 0

                s, r, done = env.reset(), 0, False
                curr_rewards = []
                
                if buffer.is_full():
                    break

        update()

        end = time.time()
        print(f"{end - start}s, \t episode: {ep}, \t return: {np.mean(ep_returns)}, \t episode length: {np.mean(ep_lens)} \t {np.exp(agent.policy.log_std.cpu().data.numpy())}")
        
        if ep - last_save >= save_freq:
            torch.save(agent.state_dict(), f"{save_path}_{ep}.pth")
            last_save = ep
        
    torch.save(agent.state_dict(), f"{save_path}_{ep}.pth")

#python ppo.py --save_freq 50 --epochs 1500 --bs 4000 --max_ep_len 1500 --discount 0.99 --lam 0.97 --eps_clip 0.2 --seed 123 --pi_lr 3e-4 --pi_update_steps 80 --v_lr 1e-3 --v_update_steps 80 --hidden_layers "[64, 32]"

if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HopperBulletEnv-v0")
    parser.add_argument("--path", type=str, default="models/hopper_model")
    parser.add_argument("--load_ep", type=int, default=-1)
    parser.add_argument("--tests", type=int, default=0)
    parser.add_argument("--init_ep", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--bs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--update_steps", type=int, default=80)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--max_ep_len", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=420690)
    parser.add_argument("--hidden_layers", type=str, default="[128, 64]")
    parser.add_argument("--init_std", type=float, default=0.5)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    args = parser.parse_args()
    print(args)

    env = gym.make(args.env)

    if args.seed > -1:
        torch.manual_seed(args.seed)
        env.seed(args.seed)
        np.random.seed(args.seed)

    hidden_layers = []
    if args.hidden_layers[1:-1].strip() != "":
        hidden_layers = [int(x) for x in args.hidden_layers[1:-1].split(",")]

    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], hidden_layers=hidden_layers, init_std=args.init_std).to(device)

    model_path = f"{os.path.dirname(__file__)}/{args.path}"

    if args.load_ep >= 0:
        agent.load_state_dict(torch.load(f"{model_path}_{args.load_ep}.pth"))

    if args.episodes > 0 and not args.test_only:
        train(agent=agent, env=env, episodes=args.episodes, batch_size=args.bs, save_path=model_path, save_freq=args.save_freq, 
            discount=args.discount, lam=args.lam, init_ep=args.init_ep, max_ep_len=args.max_ep_len, 
            eps_clip=args.eps_clip, vf_coef=args.vf_coef, lr=args.lr, update_steps=args.update_steps)

    if args.tests > 0:
        test_agent(agent, env, args.tests, 1.0 / 60.0)

    env.close()