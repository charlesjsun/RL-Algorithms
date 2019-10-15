import tensorflow as tf
import numpy as np

def gaussian_log_prob(self, actions, means, log_stds):
    return -0.5 * (tf.reduce_sum(((actions - means) / tf.exp(log_stds)) ** 2 + 2 * log_stds + np.log(2 * np.pi), axis=1))

def discounted_cumsum(self, x, discount):
    ret = np.zeros_like(x, dtype="float32")
    ret[-1] = x[-1]
    for t in range(len(x) - 2, -1, -1):
        ret[t] = x[t] + discount * ret[t + 1]
    return ret