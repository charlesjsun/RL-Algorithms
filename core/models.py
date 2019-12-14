import torch
import torch.nn as nn

import numpy as np

from core.agents import MLP

class DeterministicDeltaModel(MLP):
    """ Encodes the change in state that occurs as a result of executing a given action from given state
    """
    def __init__(self, state_dim, action_dim, hidden_layers, hidden_activation=nn.Tanh):
        super(DeterministicDeltaModel, self).__init__(
            state_dim + action_dim,
            state_dim,
            hidden_layers, hidden_activation, None
        )

    def forward(self, states, actions):
        """ Returns the predicted change in states given states and actions.
            next_states = states + self.forward(states, actions)

        Args:
            states: (n, state_dim) tensor
            actions: (n, action_dim) tensor

        Returns:
            (n, state_dim) tensor of state changes,
        """
        return self.model(torch.cat([states, actions], 1))

class RandomShootingMPCPolicy:
    """ Generate action based on random shooting on a given model. """
    def __init__(self, state_dim, action_dim, env, model, num_seqs, horizon, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.env = env
        self.model = model
        self.num_seqs = num_seqs
        self.horizon = horizon
        self.device = device
    
    def sample_action_seqs(self):
        """ Uniformly sample action sequences of horizon length, num_seqs times
        
        Returns:
            (num_seqs, horizon, action_dim) array
        """
        return np.random.rand(self.num_seqs, self.horizon, self.action_dim) * (self.action_high - self.action_low) + self.action_low

    def __call__(self, state):
        """ Returns the best action to take given the state

        Args:
            state: (state_dim,) array

        Returns:
            (action_dim,) action array,
        """
        reward_seqs = np.zeros(self.num_seqs)
        action_seqs = self.sample_action_seqs()
        states = np.repeat(np.expand_dims(state, axis=0), self.num_seqs, axis=0)
        for t in range(self.horizon):
            actions = action_seqs[:, t]

            rewards, dones = self.env.get_rewards_dones(states, actions)
            reward_seqs += rewards * dones

            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.FloatTensor(actions).to(self.device)
            states += self.model(states_t, actions_t).detach().cpu().numpy()

        best_seq = np.argmax(reward_seqs)
        return action_seqs[best_seq, 0]