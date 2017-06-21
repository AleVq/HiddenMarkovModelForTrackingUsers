import numpy as np
import pandas as pd
from HybridHMM import HybridHMM
from DataParser import parse_data


class CrossValidator:
    # data and targets file-paths, k of k-fold, choose among ann or HMM, array of value for ann: eta and runs
    def __init__(self, data_fp, targets_fp, k):
        #ds = parse_data(data_fp, targets_fp)
        #ds.to_csv(path_or_buf='../df.csv')
        ds = pd.read_csv('../df.csv', header=0, sep=';')  # TODO ripristinare il vecchio parser
        subset_size = np.math.ceil(len(ds) / k)  # size of test set for the k-fold cross val.
        results = []  # contains the percentage of non-matched prediction for each fold
        for i in range(k):
            test_set = ds.ix[i * subset_size:][:subset_size].reset_index(drop=True)
            valid_set = ds.ix[(i + 1) * subset_size:][:subset_size].reset_index(drop=True)
            if i == 0:
                training_set = ds.ix[(i + 2) * subset_size:].reset_index(drop=True)
            else:
                training_set = pd.concat([ds.ix[:i * subset_size], ds.ix[(i + 2) * subset_size:]]).reset_index(
                    drop=True)
            # hybrid model's initialization
            model = HybridHMM(training_set)
            # training and testing of the fold
            err = model.test(valid_set)
            print('Fold: ', i, 'error on validation: ', err)
            print('Fold: ', i, 'error on test: ', model.test(test_set))
            results.append(err)


def execute_my_hmm():
    cv = CrossValidator('../A_Sensors.txt', '../A_ADLs.txt', 10)


execute_my_hmm()
