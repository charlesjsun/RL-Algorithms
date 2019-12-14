from gym.envs.registration import register

register(id='HopperMBEnv-v0',
         entry_point='envs.hopper:HopperMBEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)