import tensorflow as tf

sess = tf.Session()
from keras import backend as k
from keras.models import load_model

k.set_session(sess)
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn import preprocessing


# input:
# ds = dataset from which features and targets will be extracted, pandas DataFrame
# states_count = number of possible states, int
# output:
# encoded features, pandas Dataframe
# encoded targets, numpy ndarray
def encode_features_targets(ds, states_count):
    features = ds.ix[:, :ds.shape[1] - 1]
    encoded_targets = np.zeros((ds.shape[0], states_count))
    row = 0
    for t_value in ds.ix[:, ds.shape[1] - 1]:
        encoded_targets[row, int(t_value)] = 1.0
        row = row + 1
    return features, encoded_targets


# input:
# pred = hybrid model's output states sequence, numpy ndarray
# targets = target states sequence, numpy ndarray
# output:
# error of the prediction, float
def err_rate(pred, targets):
    err = 0
    l = len(pred)
    for i in range(l):
        err += not (pred[i] == targets[i])
    return err / l


# input:
# ds = dataset, pandas DataFrame
# num_states = number of possible states, int
# output:
# HMM's transition matrix, numpy ndarray
def build_transmat(ds, num_states):
    trans_matrix = np.zeros((num_states, num_states))
    # identifying activity x-th and (x+1)th
    for x in ds.iterrows():
        if x[0] == ds.shape[0] - 2:  # the last element has no transition obviously
            break
        a1 = int(x[1].iloc[-1])
        a2 = int(ds.iloc[x[0] + 1].iloc[-1])
        trans_matrix[a1, a2] = trans_matrix[a1, a2] + 1
    # normalizing transition matrix
    return preprocessing.normalize(trans_matrix, norm='l1')


class HybridHMM:
    # input:
    # ds = training set, pandas DataFrame
    def __init__(self, ds):
        # separating features from targets
        # building one-hot encoded targets
        self.state_frequency = pd.Series.sort_index(ds.ix[:, ds.shape[1] - 1].value_counts())
        # priors prob. for hidden states
        self.pi = np.divide(self.state_frequency, ds.shape[0]).as_matrix()
        self.states_count = self.state_frequency.shape[0]
        self.features_count = ds.shape[1] - 1
        self.t = build_transmat(ds, self.states_count)

    # neural network training
    # input: training set, pandas DataFrame
    def train(self, training_set):
        features, targets = encode_features_targets(training_set, self.states_count)
        s = features.shape[0]
        X_train = features.reset_index(drop=True).as_matrix()
        Y_train = targets
        # setting model's architecture
        model = Sequential()
        model.add(Dense(self.features_count, activation='relu', input_shape=(12,)))
        model.add(Dense(2 * self.features_count, activation='relu'))
        model.add(Dense(3 * self.features_count, activation='relu'))
        model.add(Dense(2 * self.features_count, activation='relu'))
        model.add(Dense(self.states_count, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(X_train, Y_train,
                  batch_size=1440, epochs=10, verbose=0)
        model.save('my_model.h5')

    # input:
    # test_set = testing set, pandas DataFrame
    # output:
    # error of hybrid model, error of neural network, both int
    # hybrid's prediction, numpy array
    # target obs sequence, numpy ndarray
    def test(self, test_set):
        features = test_set.ix[:, :test_set.shape[1] - 1].reset_index(drop=True).as_matrix()
        targets = test_set.ix[:, test_set.shape[1] - 1].reset_index(drop=True).as_matrix()
        hybrid_prediction = np.array(self.decode(features))
        print('PREDICTION: ', hybrid_prediction)
        x, y = encode_features_targets(test_set, self.states_count)
        model = load_model('my_model.h5')
        score = model.evaluate(x.as_matrix(), y, verbose=0)
        error_rate = err_rate(hybrid_prediction, targets)
        return error_rate, score[1], np.array(hybrid_prediction), targets.astype(int)

    # input:
    # obs_seq = sequence of observations, numpy ndarray
    # output:
    # most likely sequence of states, python's list
    def decode(self, obs_seq):
        model = load_model('my_model.h5')
        forward = np.zeros((self.states_count, len(obs_seq)))
        backward = np.ones((self.states_count, len(obs_seq)), 'int32') * -1
        # initialization
        out = model.predict(np.atleast_2d([obs_seq[0]]))
        forward[:, 0] = np.squeeze(np.multiply(self.pi, out).T)
        for t in range(1, len(obs_seq)):
            out = model.predict(np.atleast_2d([obs_seq[t]]))
            forward[:, t] = (forward[:, t - 1, None].dot(out) * self.t).max(0)
            backward[:, t] = (np.tile(forward[:, t - 1, None], [1, self.states_count]) * self.t).argmax(0)
        # termination
        tokens = [forward[:, -1].argmax()]
        for i in range(len(obs_seq) - 1, 0, -1):
            tokens.append(backward[tokens[-1], i])
        return tokens[::-1]
