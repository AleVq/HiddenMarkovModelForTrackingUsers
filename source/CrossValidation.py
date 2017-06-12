import numpy as np
import pandas as pd
from HybridHMM import HybridHMM
import NeuralNetwork
from DataParser import parse_data


class CrossValidator:
    # data and targets file-paths, k of k-fold, choose among ann or HMM, array of value for ann: eta and runs
    def __init__(self, data_fp, targets_fp, k, model_choice, ann_train):
        ds = parse_data(data_fp, targets_fp)
        subset_size = np.math.ceil(len(ds) / k)
        results = pd.DataFrame([])
        for i in range(k):
            test_set = ds.ix[i * subset_size:][:subset_size]
            training_set = pd.concat([ds.ix[:i * subset_size], ds.ix[(i + 1) * subset_size:]])
            if model_choice == 'ann':
                model = NeuralNetwork()
            else:
                model = HybridHMM(training_set)
            model.train(training_set, ann_train[0], ann_train[1])
            print('ANN trained with testing set')
            #results = model.test(test_set)


def execute_my_hmm():
    cv = CrossValidator('../A_Sensors.txt', '../A_ADLs.txt', 100, 'hmm', [0.02, 5000])


execute_my_hmm()