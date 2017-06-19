import numpy as np
import pandas as pd
from HybridHMM import HybridHMM
from DataParser import parse_data


class CrossValidator:
    # data and targets file-paths, k of k-fold, choose among ann or HMM, array of value for ann: eta and runs
    def __init__(self, data_fp, targets_fp, k, ann_train):
        ds = parse_data(data_fp, targets_fp)
        subset_size = np.math.ceil(len(ds) / k)  # size of test set for the k-fold cross val.
        results = pd.Series([])  # contains the percentage of matched prediction for each fold
        state_frequency = pd.Series.sort_index(ds.ix[:, ds.shape[1] - 1].value_counts())
        for i in range(k):
            test_set = ds.ix[i * subset_size:][:subset_size].reset_index(drop=True)
            valid_set = ds.ix[(i+1) * subset_size:][:subset_size].reset_index(drop=True)
            if i==0:
                training_set = ds.ix[(i + 2) * subset_size:].reset_index(drop=True)
            else:
                training_set = pd.concat([ds.ix[:i * subset_size], ds.ix[(i + 2) * subset_size:]]).reset_index(drop=True)
            # hybrid model's initialization
            model = HybridHMM(training_set, state_frequency)
            err = 1
            # training and testing of the fold
            while err > 0.4:
                model.train(ann_train[0], ann_train[1])
                err = model.test(valid_set)
                print('Error of hybrid model: ', err)
            results.append(pd.Series([model.test(test_set)]))
        print(results)


def execute_my_hmm():
    cv = CrossValidator('../A_Sensors.txt', '../A_ADLs.txt', 10, [0.03, 10000])


execute_my_hmm()
