import tensorflow as tf

sess = tf.Session()
from keras import backend as k

k.set_session(sess)
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn import preprocessing
from TransitionMatrixGen import build_transmat


def convert_targets(self, targets):
    result = np.zeros((targets.shape[0], self.states.shape[0]))
    t = 0
    for r in result:
        r[int(targets.ix[t])] = 1
        t = t + 1
    return result


def encode_features_targets(ds):
    states_count = pd.Series.sort_index(ds.ix[:, ds.shape[1] - 1].value_counts()).shape[0]
    features = ds.ix[:, :ds.shape[1] - 1]
    encoded_targets = np.zeros((ds.shape[0], states_count))
    row = 0
    for t_value in ds.ix[:, ds.shape[1] - 1]:
        encoded_targets[row, int(t_value)] = 1.0
        row = row + 1
    return features, encoded_targets


# input: ann output and target w.r.t. 1 single example
def err_rate(pred, targets):
    err = 0
    l = len(pred)
    for i in range(l):
        err += not (pred[i] == targets[i])
    return err / l


class HybridHMM:
    # ds: pandas DataFrame that contains the ds (training set),
    # state_frequency: pandas Series that contains the frequency which they appear with
    def __init__(self, ds):
        # separating features from targets
        # building one-hot encoded targets
        features, targets = encode_features_targets(ds)
        self.labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        self.f_priors = features.groupby(
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']).size().reset_index(name='F_Priors')
        self.f_frequencies = self.f_priors['F_Priors'].divide(features.shape[0])
        del self.f_priors['F_Priors']
        self.ds = ds
        self.state_frequency = pd.Series.sort_index(ds.ix[:, ds.shape[1] - 1].value_counts())
        self.states_count = self.state_frequency.shape[0]
        # priors prob. for hidden states
        self.pi = np.divide(self.state_frequency, ds.shape[0]).as_matrix()
        self.o = np.zeros([self.states_count, self.f_priors.shape[0]])  # initialized to zero
        self.t = build_transmat(ds, self.states_count)
        # ANN initialization and training
        self.ann = self.train(features, targets)

    def train(self, features, targets):
        s = features.shape[0]
        k = s / 10
        X_train = features.ix[:int(s - k)].reset_index(drop=True).as_matrix()
        Y_train = targets[:int(targets.shape[0] - k + 1)]
        X_test = features.ix[int(s - k):].reset_index(drop=True).as_matrix()
        Y_test = targets[int(targets.shape[0] - k):]
        model = Sequential()
        model.add(Dense(self.o.shape[1], activation='relu', input_shape=(12,)))
        model.add(Dense(2 * self.o.shape[1], activation='relu'))
        model.add(Dense(3 * self.o.shape[1], activation='relu'))
        model.add(Dense(2 * self.o.shape[1], activation='relu'))
        model.add(Dense(self.states_count, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(X_train, Y_train,
                  batch_size=1332, epochs=10, verbose=0)
        score = model.evaluate(X_test, Y_test, verbose=0)
        return model

    def get_prediction(self, x):
        return self.ann.predict(x)

    def test(self, test_set, seq_length):
        features = test_set.ix[:, :test_set.shape[1] - 1].reset_index(drop=True).as_matrix()
        targets = test_set.ix[:, test_set.shape[1] - 1].reset_index(drop=True).as_matrix()
        error_rates = []
        seq_length = 60 * 24  # express in segments
        for t in range(0, features.shape[0] - seq_length, seq_length):  # 24 * 60 = seq_length duration in minutes
            hybrid_prediction = self.decode(features[t:(t + seq_length), :])
            error_rates.append(err_rate(hybrid_prediction, targets[t:(t + seq_length)]))
        error_rate_mean = np.mean(error_rates)
        return error_rate_mean

    # input: sequence of observations
    def decode(self, obs_seq):
        viterbi = np.zeros((self.states_count, len(obs_seq)))
        back_pointer = np.ones((self.states_count, len(obs_seq)), 'int32') * -1
        # initialization
        out = self.update_obs_mat(obs_seq[0])
        viterbi[:, 0] = np.squeeze(out.T)
        for t in range(1, len(obs_seq)):
            out = self.update_obs_mat(obs_seq[0])
            viterbi[:, t] = (viterbi[:, t - 1, None].dot(out) * self.t).max(0)
            back_pointer[:, t] = (np.tile(viterbi[:, t - 1, None], [1, self.states_count]) * self.t).argmax(0)
        # termination
        tokens = [viterbi[:, -1].argmax()]
        for i in range(len(obs_seq) - 1, 0, -1):
            tokens.append(back_pointer[tokens[-1], i])
        return tokens[::-1]

    # input: configuration, array of size equal to number of features
    def config_index(self, config):
        for i in self.f_priors.iterrows():
            if (i[1] == config).all():
                return i[0]
        return None

    # input: list of active features at a given time,
    def access_obs(self, obs):
        active_f = np.array(np.nonzero(obs), dtype=int)
        mul = np.ones((self.states_count,))
        for o in active_f[0]:
            mul = np.multiply(mul, self.o[:, o])
        mul = np.transpose([mul])
        return mul

    # input: observation at a given time
    def update_obs_mat(self, obs):
        ann_pred = self.get_prediction(np.atleast_2d([obs]))
        return ann_pred

    def bayes_rule(self, out, t_priors, observed_conf):
        out = np.divide(out, t_priors)
        obsmatrix_ = np.zeros((t_priors.shape[0], self.f_priors.shape[0]))
        for col in range(self.f_frequencies.shape[0]):
            obsmatrix_[:, col] = np.multiply(out, self.f_frequencies.ix[col])
        obsmatrix_ = preprocessing.normalize(obsmatrix_, norm='l1')
        for r in obsmatrix_:
            print(np.sum(r))
        return obsmatrix_
