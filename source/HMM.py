import numpy as np
import pandas as pd
import math
from hmmlearn import hmm
from NeuralNetwork import NeuralNetwork
from CrossValidation import CrossValidator


class HMM:
    def __init__(self, data_fp, targets_fp):
        self.ds = pd.read_csv(data_fp, sep='\s+', header=0, parse_dates=[['Start', 'time'], ['End', 'time.1']])
        self.target_activities = pd.read_csv(targets_fp, sep='\s+',
                                             header=0, parse_dates=[['Start', 'time'], ['End', 'time.1']])
        self.model = hmm.MultinomialHMM(n_components=self.target_activities.shape[0], algorithm='viterbi')
        self.n_features = pd.Series(self.ds.ix[:, 2].unique())
        # stati e relativa frequenza
        self.states = pd.Series(self.target_activities.ix[:,-1].unique())
        self.model.transmat_ = np.random.random([self.states.shape[0], self.states.shape[0]])
        # define obs matrix by using an ANN:
        ann = NeuralNetwork(np.array([self.n_features.shape[0], math.ceil(self.n_features.shape[0] * 1.5),
                                      math.ceil(self.n_features.shape[0] * 1.5), self.states.shape[0]]), 'softplus')
        delta_t = np.timedelta64(1, 'm')
        #  matrix which represents the dataset ready for training
        #  shape: [num of row in act matrix, num of features + target]
        ann_dataset = pd.DataFrame(np.zeros((self.target_activities.shape[0], self.n_features.shape[0])))
        f = 0 # counter to iterate over features' matrix
        a = self.ds.ix[0]['Start_time']
        for act in self.target_activities.iterrows():
            # iterate over data sensors w.r.t. the time segment of the considered activity
            while self.ds.ix[f]['Start_time'] <= act[1]['End_time.1']:# and self.ds.ix[f]['Start_time'] >= act[1]['Start_time']:
                # 1 is assigned to the elem i,j if the j-th feature w.r.t. to the i-th activity
                ann_dataset.ix[act[0], self.n_features[self.n_features == self.ds.ix[f]['Location']].index[0]] = 1
                f = f + 1
            ann_dataset.ix[act[0], -1] = self.states[self.states == act[1]['Activity']].index[0]
        print(ann_dataset)

def execute_my_hmm():
    my_hmm = HMM('../A_Sensors.txt', '../A_ADLs.txt')


execute_my_hmm()
