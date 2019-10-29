import torch
import numpy as np

from core.utils import discounted_cumsum_torch

class ReplayBuffer:
    """ Replay Buffer stores the experiences of the agent"""
    def __init__(self, max_size, state_dim, action_dim, device):
        self.max_size = int(max_size)

        self.states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=np.float32)

        self.size = 0
        self.index = 0

        self.device = device

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
            torch.FloatTensor(self.states[indices]).to(self.device).detach(),
            torch.FloatTensor(self.actions[indices]).to(self.device).detach(),
            torch.FloatTensor(self.rewards[indices]).to(self.device).detach(),
            torch.FloatTensor(self.next_states[indices]).to(self.device).detach(),
            torch.FloatTensor(self.dones[indices]).to(self.device).detach()
        )

class GAEBuffer:
    def __init__(self, batch_size, lam, discount, state_dim, action_dim, agent, device):
        self.batch_size = batch_size
        self.lam = lam
        self.discount = discount

        self.agent = agent
        self.device = device
        
        self.states = torch.zeros((batch_size, state_dim)).to(self.device)
        self.actions = torch.zeros((batch_size, action_dim)).to(self.device)
        self.rewards = torch.zeros(batch_size).to(self.device)
        self.advs = torch.zeros(batch_size).to(self.device)
        self.log_probs = torch.zeros(batch_size).to(self.device)

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
        values, log_probs, _ = self.agent.evaluate(self.states[trajectory], self.actions[trajectory])
        if values.dim() == 0:
            values = values.reshape(1)
        if log_probs.dim() == 0:
            log_probs = log_probs.reshape(1)
        values = torch.cat((values, torch.Tensor([final_val]).to(self.device)))
        rewards = torch.cat((self.rewards[trajectory], torch.Tensor([final_val]).to(self.device)))
        self.log_probs[trajectory] = log_probs

        # GAE calculations for actor update
        deltas = rewards[:-1] + self.discount * values[1:] - values[:-1]
        self.advs[trajectory] = discounted_cumsum_torch(deltas, self.discount * self.lam)

        # rewards to go calculations for critic update, estimate final returns if not terminal
        self.rewards[trajectory] = discounted_cumsum_torch(rewards, self.discount)[:-1]

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
