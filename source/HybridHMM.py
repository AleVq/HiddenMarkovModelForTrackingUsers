import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn import preprocessing

from NeuralNetwork import NeuralNetwork
from TransitionMatrixGen import build_transmat


def bayes_rule(out, t_priors, f_priors):
    scaled_likelihood = np.divide(out, t_priors)
    obsmatrix_ = np.zeros((t_priors.shape[0], f_priors.shape[1]))
    for col in range(f_priors.shape[1]):
        feature_prior = f_priors[0][col]
        obsmatrix_[:, col] = np.multiply(scaled_likelihood, feature_prior)
        obsmatrix_ = preprocessing.normalize(obsmatrix_, norm='l1')
    return obsmatrix_


def convert_targets(self, targets):
    result = np.zeros((targets.shape[0], self.states.shape[0]))
    t = 0
    for r in result:
        r[int(targets.ix[t])] = 1
        t = t + 1
    return result


def encode_features_targets(ds, states_count):
    states_count = pd.Series.sort_index(ds.ix[:, ds.shape[1] - 1].value_counts()).shape[0]
    features = ds.ix[:, :ds.shape[1] - 2]
    encoded_targets = np.zeros((ds.shape[0], states_count))
    row = 0
    for t_value in ds.ix[:, ds.shape[1] - 1]:
        encoded_targets[row, int(t_value)] = 1.0
        row = row + 1
    return features, encoded_targets


class HybridHMM:
    # ds: pandas DataFrame that contains the ds (training set),
    # state_frequency: pandas Series that contains the frequency which they appear with
    def __init__(self, ds, state_frequency):
        self.ds = ds
        self.state_frequency = state_frequency
        self.states_count = state_frequency.shape[0]
        f_priors = ds[ds >= 1.0].count()  # prior probabilities of features
        reshape_for_norm = f_priors.ix[:f_priors.shape[0]-2].values.reshape(1,-1)
        self.f_priors = preprocessing.normalize(reshape_for_norm, norm='l1')
        self.model = hmm.MultinomialHMM(n_components=self.states_count, algorithm='viterbi')
        self.model.startprob_ = np.divide(state_frequency, ds.shape[0])
        self.model.emissionprob_ = np.zeros([self.states_count, ds.shape[1]-1])  # initialized to zero
        self.model.transmat_ = build_transmat(ds, self.states_count)
        # ANN initialization
        self.ann = NeuralNetwork(ds.shape[1]-1, self.states_count)

    def train(self, learning_rate, runs):
        # separating features from targets
        # building one-hot encoded targets
        features, targets = encode_features_targets(self.ds)
        row = 0
        for t_value in self.ds.ix[:, self.ds.shape[1]-1]:
            targets[row, int(t_value)] = 1.0
            row = row + 1
        # training ANN
        for batch_run in range(0, features.shape[0], int(features.shape[0]/50)):
            print('Batched istances: ', batch_run)
            self.ann.train(features[batch_run:][:int(features.shape[0]/50)],
                           targets[batch_run:][:int(features.shape[0]/50)])

    def test(self, test_set):
        features, targets = encode_features_targets(test_set)

        # compute bayes for each time slice
        target = np.zeros([self.states_count])
        target[int(self.ds.ix[0, self.ds.shape[1]-1])] = 1
        target = [target]
        out = self.ann.get_prediction([self.ds.ix[0, :self.ds.shape[1]-2]])
        # priors prob. for hidden states
        t_priors = np.divide(self.state_frequency, self.ds.shape[0])
        t_priors = t_priors.as_matrix()
        self.model.startprob_ = t_priors
        self.model.emissionprob_ = bayes_rule(out, t_priors, self.f_priors)
        # self.model.fit(self.ds.ix[:, self.ds.shape[1] - 2], self.ds.shape[0])
        test_ds = self.ds.head(n=10).ix[:, :self.ds.shape[1]-2]
        # TODO integrate viterbi's algorithm
        dimensions = [test_ds.shape[0]]
        sample, sample_targets= self.model.sample(510)
        result = self.model.predict(test_ds.as_matrix().astype(int))
        print(result)

    # def test(self, test_set):
