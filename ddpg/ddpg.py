import torch
import torch.nn as nn
import numpy as np 
import gym
import pybullet_envs
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """ Replay Buffer stores the experiences of the agent"""
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = int(max_size)

        self.states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=np.float32)

        self.size = 0
        self.index = 0

    def store(self, s, a, r, next_s, done):
        self.states[self.index] = s
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.next_states[self.index] = next_s
        self.dones[self.index] = done
        if self.size < self.max_size:
            self.size += 1
        self.index = (self.index + 1) % self.max_size

    def sample(self, num):
        indices = np.random.randint(0, self.size, size=num)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def sample_torch(self, num):
        indices = np.random.randint(0, self.size, size=num)
        return (
            torch.FloatTensor(self.states[indices]).to(device).detach(),
            torch.FloatTensor(self.actions[indices]).to(device).detach(),
            torch.FloatTensor(self.rewards[indices]).to(device).detach(),
            torch.FloatTensor(self.next_states[indices]).to(device).detach(),
            torch.FloatTensor(self.dones[indices]).to(device).detach()
        )

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._build_policy_model([state_dim] + hidden_layers + [action_dim])

    def _build_policy_model(self, sizes):
        layers = []
        for i, o in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(i, o))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.policy = nn.Sequential(*layers)

    def forward(self, states):
        return self.policy(states)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._build_q_model([state_dim + action_dim] + hidden_layers + [1])

    def _build_q_model(self, sizes):
        layers = []
        for i, o in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(i, o))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.qf = nn.Sequential(*layers)

    def forward(self, states, actions):
        return self.qf(torch.cat([states, actions], 1))

class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, action_noise, action_low, action_high, hidden_layers=[32, 32]):
        super(Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_noise = action_noise
        self.action_low = action_low
        self.action_high = action_high
        self.hidden_layers = hidden_layers
        self.actor = Actor(state_dim, action_dim, hidden_layers)
        self.critic = Critic(state_dim, action_dim, hidden_layers)

    def forward(self):
        raise NotImplementedError

    def evaluate(self, states, actions):
        """ Returns the q-values of the given states and actions

        Args:
            states: (n, state_dim) tensor
            actions: (n, action_dim) tensor

        Returns:
            (n,) tensor of q-values
        """
        return torch.squeeze(self.critic(states, actions))

    def evaluate_states(self, states):
        """ Returns the actions taken given the current states

        Args:
            states: (n, state_dim) tensor

        Returns:
            (n, action_dim) tensor
        """
        return self.actor(states)

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
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy()
        noise = np.random.normal(0, self.action_noise, size=self.action_dim)
        return np.clip(action + noise, self.action_low, self.action_high)

def train(agent=None, env=None, episodes=10000, buffer_size=1e6, batch_size=100, save_path=None, save_freq=100, init_ep=0, 
        discount=0.99, max_ep_len=1000, lr=1e-3, polyak=0.995, min_steps_update=500):

    target = Agent(agent.state_dim, agent.action_dim, agent.action_noise, 
                agent.action_low, agent.action_high, agent.hidden_layers).to(device)
    target.load_state_dict(agent.state_dict())

    buffer = ReplayBuffer(buffer_size, agent.state_dim, agent.action_dim)
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=lr)

    def update(update_steps):
        for _ in range(update_steps):
            states, actions, rewards, next_states, dones = buffer.sample_torch(batch_size)
            next_actions = agent.evaluate_states(next_states)

            # q-function loss
            target_qs = rewards + discount * (1.0 - dones) * agent.evaluate(next_states, next_actions)
            critic_loss = torch.mean((agent.evaluate(states, actions) - target_qs) ** 2)
            
            # optimize q one step
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # policy loss
            actor_loss = -torch.mean(agent.evaluate(states, agent.evaluate_states(states)))

            # optimize pi one step
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # update target networks
            for param, target_param in zip(agent.parameters(), target.parameters()):
                target_param.data.copy_(polyak * target_param.data + (1.0 - polyak) * param.data)

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
            a = agent.sample_action_numpy(s)
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

                if sum(ep_lens) >= min_steps_update:
                    break

        update(sum(ep_lens))

        end = time.time()
        print(f"{end - start}s, \t episode: {ep}, \t return: {np.mean(ep_returns)}, \t episode length: {np.mean(ep_lens)}")
        
        if ep - last_save >= save_freq:
            torch.save(agent.state_dict(), f"./{save_path}_{ep}.pth")
            last_save = ep
        
    torch.save(agent.state_dict(), f"./{save_path}_{ep}.pth")

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
    parser.add_argument("--env", type=str, default="HopperBulletEnv-v0")
    parser.add_argument("--path", type=str, default="models/hopper_model")
    parser.add_argument("--load_ep", type=int, default=-1)
    parser.add_argument("--tests", type=int, default=0)
    parser.add_argument("--init_ep", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--bs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--max_ep_len", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42069)
    parser.add_argument("--hidden_layers", type=str, default="[128, 64]")
    parser.add_argument("--action_noise", type=float, default=0.1)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--min_steps_update", type=int, default=500)
    parser.add_argument("--buffer_size", type=int, default=1e6)
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

    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0], args.action_noise,
                    env.action_space.low, env.action_space.high, hidden_layers=hidden_layers).to(device)

    if args.load_ep >= 0:
        agent.load_state_dict(torch.load(f"./{args.path}_{args.load_ep}.pth"))

    if args.episodes > 0 and not args.test_only:
        train(agent=agent, env=env, episodes=args.episodes, batch_size=args.bs, save_path=args.path, save_freq=args.save_freq, 
            discount=args.discount, init_ep=args.init_ep, max_ep_len=args.max_ep_len, lr=args.lr, polyak=args.polyak,
            min_steps_update=args.min_steps_update, buffer_size=args.buffer_size)

    if args.tests > 0:
        test_agent(agent, env, args.tests, 0.025)

    env.close()