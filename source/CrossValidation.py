import numpy as np
import pandas as pd
from HybridHMM import HybridHMM
from DataParser import parse_data
import os.path
import matplotlib.pyplot as plt


class CrossValidator:
    # input: data and targets file-paths, k of k-fold,
    # input: length of observations' sequence for viterbi alg
    def __init__(self, data_fp, targets_fp, k, fp):
        if not(os.path.isfile(fp)):
            temp, states_label = parse_data(data_fp, targets_fp)
            temp.to_csv(path_or_buf=fp, index=False, sep=';')
        ds = pd.read_csv(fp, header=0, sep=';')
        subset_size = np.math.ceil(len(ds) / k)  # size of test set for the k-fold cross val.
        results = []  # contains the percentage of non-matched prediction for each fold
        nn_accuracies = []
        for i in range(k):
            test_set = ds.ix[i * subset_size:][:subset_size].reset_index(drop=True)
            if i == 0:
                training_set = ds.ix[(i + 2) * subset_size:].reset_index(drop=True)
            else:
                training_set = pd.concat([ds.ix[:i * subset_size],
                                          ds.ix[(i + 2) * subset_size:]]).reset_index(drop=True)
            training_set = training_set.drop(training_set.index[:subset_size]).reset_index(drop=True)
            # hybrid model's initialization, training and testing
            model = HybridHMM(training_set)
            model.train(training_set)
            test_err, nn_accuracy, prediction, targets = model.test(test_set)
            test = pd.DataFrame(np.vstack((prediction, targets)).T)  # getting prediction and targets together
            fig = test.plot().get_figure()
            fig.savefig('/Users/ale/Desktop/plot_test.pdf')
            # test.to_csv(path_or_buf='../comparison.csv', index=False, sep=',')
            print('Fold: ', i, 'error on test: ', test_err, ' err of ann: ', (1 - nn_accuracy))
            results.append(test_err)
            nn_accuracies.append(nn_accuracy)

        print('Average error on tests among k-fold: ', np.mean(np.array(results)))
        print('Average error of nn on tests among k-fold: ', (1 - np.mean(np.array(nn_accuracies))))
        pd.DataFrame(results).to_csv(path_or_buf='../results.csv', index=False, sep=';')
        pd.DataFrame(nn_accuracies).to_csv(path_or_buf='../NNresults.csv', index=False, sep=';')


def execute_my_hmm():
    print('Results on dataset A:')
    cv = CrossValidator('../A_Sensors.txt', '../A_ADLs.txt', 14, '../df.csv')
    print('Results on dataset B:')
    cv = CrossValidator('../B_Sensors.txt', '../B_ADLs.txt', 21, '../df2.csv')


execute_my_hmm()
