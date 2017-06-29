import numpy as np
import pandas as pd
from HybridHMM import HybridHMM
from DataParser import parse_data
import os.path


class CrossValidator:
    # input: data and targets file-paths, k of k-fold,
    # input: length of observations' sequence for viterbi alg
    def __init__(self, data_fp, targets_fp, k, seq_len, fp):
        temp = parse_data(data_fp, targets_fp)
        if not(os.path.isfile(fp)):
            temp = parse_data(data_fp, targets_fp)
            temp.to_csv(path_or_buf='../df2.csv', index=False, sep=',')
        ds = pd.read_csv('../df2.csv', header=0, sep=',')  # TODO ripristinare il vecchio parser
        subset_size = np.math.ceil(len(ds) / k)  # size of test set for the k-fold cross val.
        results = []  # contains the percentage of non-matched prediction for each fold
        for i in range(k):
            test_set = ds.ix[i * subset_size:][:subset_size].reset_index(drop=True)
            valid_set = ds.ix[(i + 1) * subset_size:][:subset_size].reset_index(drop=True)
            if i == 0:
                training_set = ds.ix[(i + 2) * subset_size:].reset_index(drop=True)
            else:
                training_set = pd.concat([ds.ix[:i * subset_size],
                                          ds.ix[(i + 2) * subset_size:]]).reset_index(drop=True)
            # hybrid model's initialization
            model = HybridHMM(training_set)
            # training and testing of the fold
            valid_err = model.test(valid_set, seq_len)
            print('Fold: ', i, 'error on validation: ', valid_err)
            test_err = model.test(test_set, seq_len)
            print('Fold: ', i, 'error on test: ', test_err)
            results.append(test_err)
        print('Average error on tests among k-fold: ', np.mean(np.array(results)))


def execute_my_hmm():
    print('Results on dataset A:')
    cv = CrossValidator('../A_Sensors.txt', '../A_ADLs.txt', 10, 500, '../df.csv')
    print('Results on dataset B:')
    cv = CrossValidator('../B_Sensors.txt', '../B_ADLs.txt', 10, 500, '../df2.csv')


execute_my_hmm()
