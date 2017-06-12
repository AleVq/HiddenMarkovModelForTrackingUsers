import numpy as np
import pandas as pd
import math
from hmmlearn import hmm
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt


def convert_targets(self, targets):
    result = np.zeros((targets.shape[0], self.states.shape[0]))
    t = 0
    for r in result:
        r[int(targets.ix[t])] = 1
        t = t + 1
    return result


class HybridHMM:
    def __init__(self, ds):
        self.ds = ds
        num_state_values = int(max(ds.ix[:, ds.shape[1]-1]) + 1)
        self.model = hmm.MultinomialHMM(n_components=ds.shape[0], algorithm='viterbi')
        self.model.transmat_ = np.random.random([num_state_values, num_state_values])
        # define obs matrix by using an self.ann:
        self.ann = NeuralNetwork(np.array([ds.shape[1]-1, math.ceil(ds.shape[1] * 1.5),
                                           math.floor(ds.shape[1] * 1.5), num_state_values]))

    def train(self, training_set, learning_rate, runs):
        # separating features from targets
        a = training_set.ix[:, :training_set.shape[1]-2]
        self.ann.train(training_set.ix[:, :training_set.shape[1]-2], training_set.ix[:, training_set.shape[1]-1], learning_rate, runs)
        print('n')

