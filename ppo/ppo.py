import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np 
import gym
import pybullet_envs
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def discounted_cumsum(x, discount):
    ret = torch.zeros_like(x)
    ret[-1] = x[-1]
    for t in range(len(x) - 2, -1, -1):
        ret[t] = x[t] + discount * ret[t + 1]
    return ret

class GAEBuffer:
    def __init__(self, batch_size, lam, discount, state_dim, action_dim):
        self.batch_size = batch_size
        self.lam = lam
        self.discount = discount
        
        self.states = torch.zeros((batch_size, state_dim)).to(device)
        self.actions = torch.zeros((batch_size, action_dim)).to(device)
        self.rewards = torch.zeros(batch_size).to(device)
        self.advs = torch.zeros(batch_size).to(device)
        self.log_probs = torch.zeros(batch_size).to(device)

        self.num = 0
        self.trajectory_start = 0

    def store_numpy(self, s, a, r):
        self.store(torch.tensor(s), torch.tensor(a), r)

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
        values, log_probs, _ = agent.evaluate(self.states[trajectory], self.actions[trajectory])
        if values.dim() == 0:
            values = values.reshape(1)
        if log_probs.dim() == 0:
            log_probs = log_probs.reshape(1)
        values = torch.cat((values, torch.Tensor([final_val]).to(device)))
        rewards = torch.cat((self.rewards[trajectory], torch.Tensor([final_val]).to(device)))
        self.log_probs[trajectory] = log_probs

        # GAE calculations for actor update
        deltas = rewards[:-1] + self.discount * values[1:] - values[:-1]
        self.advs[trajectory] = discounted_cumsum(deltas, self.discount * self.lam)

        # rewards to go calculations for critic update, estimate final returns if not terminal
        self.rewards[trajectory] = discounted_cumsum(rewards, self.discount)[:-1]

        # start new trajectory
        self.trajectory_start = self.num
    
    def batch_normalize_advs(self, eps=1e-8):
        assert self.is_full(), "Tried to batch normalize before buffer is full"
        advs_mean, advs_std = torch.mean(self.advs), torch.std(self.advs)
        self.advs = (self.advs - advs_mean) / (advs_std + eps)

    def get_buffer(self):
        assert self.is_full(), "Tried to get buffer before buffer is full"
        self.batch_normalize_advs()
        self.clear()
        return self.states.detach(), self.actions.detach(), self.rewards.detach(), self.advs.detach(), self.log_probs.detach()

    def clear(self):
        self.num = 0
        self.trajectory_start = 0

class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, init_std=0.5, hidden_layers=[32, 32]):
        super(Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._build_policy_model([state_dim] + hidden_layers + [action_dim], init_std)
        self._build_value_model([state_dim] + hidden_layers + [1])

    def _build_policy_model(self, sizes, init_std):
        layers = []
        for i, o in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(i, o))
            layers.append(nn.Tanh())
        self.policy = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((sizes[-1],), np.log(init_std)))

    def _build_value_model(self, sizes):
        layers = []
        for i, o in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(i, o))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.vf = nn.Sequential(*layers)

    def forward(self):
        raise NotImplementedError

    def evaluate(self, states, actions):
        """ Returns the state-values and log probs of the given states and actions

        Args:
            states: (n, state_dim) tensor
            actions: (n, action_dim) tensor

        Returns:
            (n,) tensor of state values,
            (n,) tensor of log probs
        """
        means = self.policy(states)
        log_stds = self.log_std.expand_as(means)
        cov = torch.diag_embed(torch.exp(log_stds * 2.0)).to(device)

        dist = MultivariateNormal(means, cov)
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
        mean = self.policy(state)
        cov = torch.diag(torch.exp(self.log_std * 2.0)).to(device)
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

    buffer = GAEBuffer(batch_size, lam, discount, agent.state_dim, agent.action_dim)
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
        print(f"{end - start}s, \t episode: {ep}, \t return: {np.mean(ep_returns)}, \t episode length: {np.mean(ep_lens)} \t {np.exp(agent.log_std.cpu().data.numpy())}")
        
        if ep - last_save >= save_freq:
            torch.save(agent.state_dict(), f"./{save_path}_{ep}.pth")
            last_save = ep
        
    torch.save(agent.state_dict(), f"./{save_path}_{ep}.pth")

def test_agent(agent, env, n_tests, delay=1.0, bullet=True):
    agent.log_std.data = torch.full((agent.action_dim,), np.log(0.1))
    for test in range(n_tests):
        if bullet:
            env.render(mode="human")
        s = env.reset()
        done = False
        total_reward = 0
        print(f"Test #{test}")
        while True:
            # time.sleep(delay)
            if bullet:
                env.camera_adjust()
            else:
                env.render()
            a = agent.sample_action_numpy(s)
            # print(f"Chose action {a} for state {s}")
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

    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0], hidden_layers=hidden_layers, init_std=args.init_std).to(device)

    if args.load_ep >= 0:
        agent.load_state_dict(torch.load(f"./{args.path}_{args.load_ep}.pth"))

    if args.episodes > 0 and not args.test_only:
        train(agent=agent, env=env, episodes=args.episodes, batch_size=args.bs, save_path=args.path, save_freq=args.save_freq, 
            discount=args.discount, lam=args.lam, init_ep=args.init_ep, max_ep_len=args.max_ep_len, 
            eps_clip=args.eps_clip, vf_coef=args.vf_coef, lr=args.lr, update_steps=args.update_steps)

    if args.tests > 0:
        test_agent(agent, env, args.tests, 1.0 / 60.0)

    env.close()