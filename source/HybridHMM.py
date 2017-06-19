import numpy as np
import pandas as pd
from hmmlearn import hmm
import math
from sklearn import preprocessing, metrics
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


def encode_features_targets(ds):
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
        # normalize features' prior probabilities
        reshape_for_norm = f_priors.ix[:f_priors.shape[0]-2].values.reshape(1,-1)
        self.f_priors = preprocessing.normalize(reshape_for_norm, norm='l1')
        self.pi = np.divide(state_frequency, ds.shape[0])
        self.o = np.zeros([self.states_count, ds.shape[1]-1])  # initialized to zero
        self.t = build_transmat(ds, self.states_count)
        # ANN initialization
        self.ann = NeuralNetwork(ds.shape[1]-1, self.states_count)

    def train(self, learning_rate, runs):
        # separating features from targets
        # building one-hot encoded targets
        features, targets = encode_features_targets(self.ds)
        # training ANN
        acc = []
        div = 20
        #a = self.ann.train(features, targets)
        d = self.ann.train(features.ix[136:456, :], targets[136:457, :])
        evs = []
        for i in range(136, 160, 1):
            pred = self.ann.get_prediction([features.ix[i, :]])[0]
            target_pred = targets[i, :]
            print(targets[i,:], pred)
            evs.append(self.evaluate(pred, target_pred))
        print(evs)
        '''
        for batch_run in range(0, features.shape[0], int(features.shape[0]/div)):
            ac_acc = (self.ann.train(features[batch_run:][:int(features.shape[0]/div)],
                           targets[batch_run:][:int(features.shape[0]/div)]))
            acc.append(ac_acc)
            if batch_run > features.shape[0]-(2*(features.shape[0]/div)):
                m = np.mean(acc)
                if m < 0.8:
                    batch_run = 0
                print('Average accuracy during NN training: ', m)
        '''
    def test(self, test_set):
        features, targets = encode_features_targets(test_set)
        features = features.reset_index(drop=True)
        features = features.as_matrix()
        # compute bayes for each time slice
        # priors prob. for hidden states
        t_priors = np.divide(self.state_frequency, self.ds.shape[0])
        t_priors = t_priors.as_matrix()
        self.pi = t_priors
        result = self.viterbi(np.arange(self.states_count), features[67:77])
        evaluation = self.evaluate(result, self.ds.reset_index(drop=True).ix[67:77, self.ds.shape[1]-1].as_matrix())
        return evaluation

    # o_space: observations' space, s_space: states' space, seq: obs. sequence
    # all input must be np.array
    # output most likely sequence
    def viterbi(self, s_space, seq):
        T1 = np.zeros((s_space.shape[0], seq.shape[0]))
        T2 = np.zeros((s_space.shape[0], seq.shape[0]))
        for t_i in range(seq.shape[0]):
            print('features: ', seq[t_i], 'predizione: ', self.ann.get_prediction([seq[t_i]]))
            for s in range(s_space.shape[0]):
                # o gets updated at each time slice:
                updated_o = bayes_rule(self.ann.get_prediction([seq[t_i]]), self.pi, self.f_priors)
                obs_indices = np.nonzero(seq[0])
                mul_obs_prob = 1
                for i in obs_indices[0]:  # t_i are independent: their joint probs
                    mul_obs_prob *= updated_o[s, i]   # are simple given by the multiplication of each one prob
                if t_i == 0:
                    T1[s, 0] = self.pi[s] * mul_obs_prob
                    T2[s, 0] = 0
                else:
                    pw_multiplied = np.multiply(T1[:, t_i-1], self.t[:, s])  # point-wise multiply
                    max_trans_prob = pw_multiplied[pw_multiplied.argmax()]
                    T1[s, t_i] = mul_obs_prob * max_trans_prob
                    T2[s, t_i] = max_trans_prob
        z = np.zeros((T1.shape[1]))
        for t in range(T1.shape[1]):
            z[t] = np.argmax(T1[:, t])
        result = np.zeros(z.shape[0])
        for t_z in range(z.shape[0]):
            result[t_z] = z[t_z]  # z contains states indices
        for i in range(z.shape[0]-1, 1, 1):
            z[i-1] = T2[z[i], i]
            result[i-1] = s[z[i-1]]
        return result

    # input: ann output and target w.r.t. 1 single example
    @staticmethod
    def evaluate(pred, targets):
        err = 0
        for i in range(pred.shape[0]):
            err += not(pred[i] == targets[i])
        return err / pred.shape[0]

