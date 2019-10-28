import torch
import numpy as np

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