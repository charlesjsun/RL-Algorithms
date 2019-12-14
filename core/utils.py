import torch
import numpy as np 

import time

def discounted_cumsum_torch(x, discount):
    ret = torch.zeros_like(x)
    ret[-1] = x[-1]
    for t in range(len(x) - 2, -1, -1):
        ret[t] = x[t] + discount * ret[t + 1]
    return ret

def test_agent(agent, env, n_tests, delay=1.0, bullet=True):
    agent.action_noise = 0.0
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
            s, reward, done, progress = env.step(a)
            total_reward += reward
            if done:
                print(f"Done. Total Reward = {total_reward}")
                time.sleep(2)
                break