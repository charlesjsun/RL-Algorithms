# RL-Algorithms

This deep reinforcement learning library provides an easy way to implement and experiment with deep RL algoirthms.

Written in PyTorch, the `core` module includes the infrastructure behind most algoirthms like
- Abstract base classes for agents
- Easy instantiation of MLP
- Q-Functions, Value Functions
- Various Policies (Gaussian, Deterministic, Categorical)
- Replay Buffer and GAE Buffer
- MLP Dynamics Model (WIP) 

Currently implemented algorithms include:
- Tabular Q-Learning
- VPG (Vanilla Policy Gradient)
- PPO (Proximal Policy Optimization)
- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed DDPG)

WIP:
- Model-Based MPC
- Model-Ensemble PPO
