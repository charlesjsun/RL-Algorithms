import torch
import torch.nn as nn

class MLPModel(nn.Module):
    """ Base class for a Multi-Layered Perceptron model. """
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_activation=nn.Tanh, final_activation=nn.Tanh):
        super(MLPModel, self).__init__()
        self._build_model([input_dim] + hidden_layers + [output_dim], hidden_activation, final_activation)

    def _build_model(self, sizes, hidden_activation, final_activation):
        layers = []
        for i, o in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(i, o))
            if hidden_activation:
                layers.append(hidden_activation())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        if final_activation:
            layers.append(final_activation())
        self.model = nn.Sequential(*layers)

    def forward(self, *input):
        raise NotImplementedError

class GaussianPolicy(MLPModel):
    """ A stocastic Gaussian policy which uses a diagonal gaussian distribution.
        Contains an attribute for the state-independent log-standard deviation which can also be learned.
    """
    def __init__(self, state_dim, action_dim, hidden_layers, init_std=0.5, hidden_activation=nn.Tanh, final_activation=nn.Tanh):
        super(GaussianPolicy, self).__init__(
            state_dim, 
            action_dim, 
            hidden_layers, hidden_activation, final_activation
        )
        self.log_std = nn.Parameter(torch.full((sizes[-1],), np.log(init_std)))

    def forward(self, states):
        """ Returns the mean and covariance matrix of the given states,
            Typically passed into a MultivariateNormal object.
        
        Args:
            states: (n, state_dim) or (state_dim,) tensor

        Returns:
            (n, action_dim) or (action_dim) means tensor,
            (n, action_dim, action_dim) or (action_dim, action_dim) covariances tensor
        """
        means = self.model(states)
        log_stds = self.log_std.expand_as(means)
        covs = torch.diag_embed(torch.exp(log_stds * 2.0))
        return means, covs

class DeterministicPolicy(MLPModel):
    """ A deterministic policy for continuous which maps states directly to actions. """
    def __init__(self, state_dim, action_dim, hidden_layers, hidden_activation=nn.Tanh, final_activation=nn.Tanh):
        super(DeterministicPolicy, self).__init__(
            state_dim, 
            action_dim, 
            hidden_layers, hidden_activation, final_activation
        )

    def forward(self, states):
        """ Returns the mean actions of the given states.

        Args:
            states: (n, state_dim) or (state_dim,) tensor

        Returns:
            (n, action_dim) or (action_dim) action tensor,
        """
        return self.model(states)

class ValueFunction(MLPModel):
    """ Approximation of the state-value function. """
    def __init__(self, state_dim, hidden_layers, hidden_activation=nn.Tanh):
        super(ValueFunction, self).__init__(
            state_dim, 
            1, 
            hidden_layers, hidden_activation, None
        )
    
    def forward(self, states):
        """ Returns the state-value given states.

        Args:
            states: (n, state_dim) or (state_dim,) tensor

        Returns:
            (n, 1) or (1,) tensor of state-values,
        """
        return self.model(states)

class QFunction(MLPModel):
    """ Approximation of the Q-value (state-action value) function. """
    def __init__(self, state_dim, action_dim, hidden_layers, hidden_activation=nn.Tanh):
        super(QFunction, self).__init__(
            state_dim + action_dim, 
            1, 
            hidden_layers, hidden_activation, None
        )

    def forward(self, states, actions):
        """ Returns the Q-value given states and actions.

        Args:
            states: (n, state_dim) or (state_dim,) tensor
            actions: (n, state_dim) or (state_dim,) tensor

        Returns:
            (n, 1) or (1,) tensor of state-values,
        """
        return self.model(torch.cat([states, actions], 1))

class Agent(nn.Module):
    """ Base class for a standard RL agent.

        Typically, subclasses will have some kind of policy model and some critic model as attributes,
        and implement the evaluate and sample methods for learning.
    """
    def __init__(self, state_dim, action_dim):
        super(Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self):
        """ Typically not implemented in subclasses. """
        raise NotImplementedError

    def evaluate(self, states, actions):
        """ Evaluate the given states and actions and return some metric

        Args:
            states: (n, state_dim) tensor
            actions: (n, action_dim) tensor

        Returns:
            Depends on implementation
        """
        raise NotImplementedError

    def evaluate_states(self, states):
        """ Evaluate the given states and return some metric

        Args:
            states: (n, state_dim) tensor

        Returns:
            Depends on implementation
        """
        raise NotImplementedError

    def sample_action(self, state):
        """ Returns an action (in torch.tensor) based on the current policy given state

        Args:
            state: (state_dim,) tensor

        Returns:
            (action_dim,) tensor
        """
        return NotImplementedError

    def sample_action_numpy(self, state):
        """ Returns an action (in np.array) based on the current policy given state

        Args:
            state: (state_dim,) numpy array

        Returns:
            (action_dim,) numpy array
        """
        raise NotImplementedError