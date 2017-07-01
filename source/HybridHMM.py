import tensorflow as tf

sess = tf.Session()
from keras import backend as k
from keras.models import load_model

k.set_session(sess)
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from TransitionMatrixGen import build_transmat


def encode_features_targets(ds, states_count):
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
        self.state_frequency = pd.Series.sort_index(ds.ix[:, ds.shape[1] - 1].value_counts())
        # priors prob. for hidden states
        self.pi = np.divide(self.state_frequency, ds.shape[0]).as_matrix()
        self.states_count = self.state_frequency.shape[0]
        self.o = np.zeros([self.states_count, ds.shape[1]-1])  # initialized to zero
        self.t = build_transmat(ds, self.states_count)

    # neural network training
    # input: training set
    def train(self, training_set):
        features, targets = encode_features_targets(training_set, self.states_count)
        s = features.shape[0]
        k = s / 10
        X_train = features.reset_index(drop=True).as_matrix()
        Y_train = targets
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
                  batch_size=1440, epochs=10, verbose=0)
        model.save('my_model.h5')

    def test(self, test_set):
        features = test_set.ix[:, :test_set.shape[1] - 1].reset_index(drop=True).as_matrix()
        targets = test_set.ix[:, test_set.shape[1] - 1].reset_index(drop=True).as_matrix()
        hybrid_prediction = np.array(self.decode(features))
        print('PREDICTION: ', hybrid_prediction)
        x, y = encode_features_targets(test_set, self.states_count)
        model = load_model('my_model.h5')
        score = model.evaluate(x.as_matrix(), y, verbose=0)
        error_rate = err_rate(hybrid_prediction, targets)
        return error_rate, score[1], np.array(hybrid_prediction), targets

    # input: sequence of observations
    # output: most likely sequence of states
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
