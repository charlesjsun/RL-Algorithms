import torch
import torch.nn as nn
import numpy as np 
import gym
import pybullet_envs
import time

import envs

from core.agents import Agent
from core.models import DeterministicDeltaModel
from core.models import RandomShootingMPCPolicy

from core.buffers import ReplayBuffer

from core.utils import test_agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MBRLAgent(Agent):
    def __init__(self, state_dim, action_dim, env, num_seqs, horizon, hidden_layers):
        super().__init__(state_dim, action_dim)
        self.hidden_layers = hidden_layers
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DeterministicDeltaModel(state_dim, action_dim, hidden_layers)
        self.actor = RandomShootingMPCPolicy(state_dim, action_dim, env, self.model, num_seqs, horizon, device)
        
    def evaluate_model(self, states, actions):
        """ Returns Q-values of critic1 of the given states and actions

        Args:
            states: (n, state_dim) tensor
            actions: (n, action_dim) tensor

        Returns:
            (n, state_dim) tensor of state_changes
        """
        return self.model(states, actions)

    def sample_action(self, state):
        """ Returns an action based on the current policy given state

        Args:
            state: (state_dim,) tensor

        Returns:
            (action_dim,) tensor
        """
        return torch.FloatTensor(self.sample_action_numpy(state.cpu().data.numpy())).to(device)

    def sample_action_numpy(self, state):
        """ Returns an action based on the current policy given state

        Args:
            state: (state_dim,) numpy array

        Returns:
            (action_dim,) numpy array
        """
        return self.actor(state) 

def train(agent, env, params):
    episodes     = params["episodes"]
    buffer_size  = params["buffer_size"]
    batch_size   = params["batch_size"]
    save_path    = params["save_path"]
    save_freq    = params["save_freq"]
    init_ep      = params["init_ep"]
    max_ep_len   = params["max_ep_len"]
    lr           = params["learning_rate"]
    start_steps  = params["start_steps"]
    update_steps = params["update_steps"]
    sample_size  = params["sample_size"]

    model_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size, agent.state_dim, agent.action_dim, device)

    def update():
        for _ in range(update_steps):
            states, actions, _, next_states, _ = buffer.sample_torch(sample_size)

            state_changes = next_states - states
            predicted_state_changes = agent.evaluate_model(states, actions)
            squared_norms = torch.sum((state_changes - predicted_state_changes) ** 2, 1)
            model_loss = torch.mean(squared_norms)
            
            model_optimizer.zero_grad()
            model_loss.backward()
            model_optimizer.step()

    # Random exploration at the beginning for start_steps
    print(f"Start steps: {int(start_steps)}")
    s, r, done = env.reset(), 0, False
    ep_len = 0
    for _ in range(int(start_steps)):
        a = env.action_space.sample()
        new_s, r, done, _ = env.step(a)
        buffer.store(s, a, r, new_s, done)
        s = new_s
        ep_len += 1
        if done or ep_len == max_ep_len:
            ep_len = 0
            s, r, done = env.reset(), 0, False

    # Actual training
    print("Start Training")
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
            start_action_time = time.time()
            a = agent.sample_action_numpy(s)
            print(time.time() - start_action_time)
            new_s, r, done, _ = env.step(a)
            buffer.store(s, a, r, new_s, done)
            s = new_s

            ep_len += 1
            curr_rewards.append(r)
        
            if done or ep_len == max_ep_len:
                ep_returns.append(sum(curr_rewards))
                ep_lens.append(ep_len)

                ep += 1
                ep_len = 0

                s, r, done = env.reset(), 0, False
                curr_rewards = []

                if sum(ep_lens) >= batch_size:
                    break

        update()

        end = time.time()
        print(f"{end - start}s, \t episode: {ep}, \t return: {np.mean(ep_returns)}, \t episode length: {np.mean(ep_lens)}")
        
        if ep - last_save >= save_freq:
            torch.save(agent.state_dict(), f"{save_path}/ep_{ep}.pth")
            last_save = ep
        
    torch.save(agent.state_dict(), f"{save_path}/ep_{ep}.pth")

if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "-e", type=str, default="HopperMBEnv-v0")
    parser.add_argument("--local_path", "-p", type=str, default="models/hopper_models")
    parser.add_argument("--seed", "-s", type=int, default=69420)
    
    parser.add_argument("--tests", type=int, default=0)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--bullet", action="store_true")

    parser.add_argument("--load_ep", type=int, default=-1)
    parser.add_argument("--init_ep", type=int, default=0)
    parser.add_argument("--save_freq", "-sf", type=int, default=10)
    
    parser.add_argument("--episodes", "-eps", type=int, default=100)
    parser.add_argument("--max_ep_len", type=int, default=2000)
    parser.add_argument("--start_steps", type=int, default=20000)
    parser.add_argument("--batch_size", "-bs", type=int, default=8000)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--update_steps", type=int, default=80)
    parser.add_argument("--sample_size", type=int, default=512)

    parser.add_argument("--mpc_horizon", type=int, default=10)
    parser.add_argument("--mpc_num_seqs", type=int, default=1000)
    parser.add_argument("--hidden_layers", "-hl", type=str, default="[250, 250]")

    args = parser.parse_args()
    params = vars(args)
    print(args)

    # setup environment
    env = gym.make(args.env_name)

    if args.seed > -1:
        torch.manual_seed(args.seed)
        env.seed(args.seed)
        np.random.seed(args.seed)

    # setup save paths
    model_path = f"{os.path.dirname(__file__)}/{args.local_path}"
    params["save_path"] = model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # setup agent
    hidden_layers = []
    if args.hidden_layers[1:-1].strip() != "":
        hidden_layers = [int(x) for x in args.hidden_layers[1:-1].split(",")]

    agent = MBRLAgent(env.observation_space.shape[0], env.action_space.shape[0], env, 
                    args.mpc_horizon, args.mpc_num_seqs, hidden_layers=hidden_layers).to(device)

    if args.load_ep >= 0:
        agent.load_state_dict(torch.load(f"{model_path}/ep_{args.load_ep}.pth"))

    # training
    if args.episodes > 0 and not args.test_only:
        train(agent, env, params)

    # testing
    if args.tests > 0:
        test_agent(agent, env, args.tests, 1.0 / 60.0, bullet=args.bullet)

    env.close()