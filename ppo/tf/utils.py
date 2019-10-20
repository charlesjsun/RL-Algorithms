import tensorflow as tf
import numpy as np
import scipy.signal

def gaussian_log_prob(actions, means, log_stds, eps=1e-8):
    return -0.5 * (tf.reduce_sum(((actions - means) / (tf.exp(log_stds) + eps)) ** 2.0 + 2.0 * log_stds + np.log(2.0 * np.pi), axis=1))

def gaussian_entropy(log_stds):
    return tf.reduce_sum(log_stds + 0.5 * np.log(2.0 * np.pi * np.e), axis=1)

def discounted_cumsum(x, discount):
    # ret = np.zeros_like(x, dtype="float32")
    # ret[-1] = x[-1]
    # for t in range(len(x) - 2, -1, -1):
    #     ret[t] = x[t] + discount * ret[t + 1]
    # return ret
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]