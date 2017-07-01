import numpy as np
import pandas as pd
from HybridHMM import HybridHMM
from DataParser import parse_data
import os.path
import math


def reverse(a):
    return 1-a


class CrossValidator:
    # input: data and targets file-paths, k of k-fold,
    # input: length of observations' sequence for viterbi alg
    # input: size of test set for the k-fold cross val.
    def __init__(self, data_fp, targets_fp, subset_size, fp, ds_label):
        if not(os.path.isfile(fp)):
            temp, states_labels, features_labels = parse_data(data_fp, targets_fp)
            temp.to_csv(path_or_buf=fp, index=False, sep=';')
            pd.DataFrame(states_labels).to_csv(path_or_buf='../labels_states_%s.csv' %ds_label, sep=',')
            pd.DataFrame(features_labels).to_csv(path_or_buf='../labels_features_%s.csv' %ds_label, sep=',')
        ds = pd.read_csv(fp, header=0, sep=';')
        k = math.floor(len(ds) / subset_size)
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
            test.columns = ['predictions', 'targets']
            # printing just 3 examples diagrams of predictions vs targets
            if i <3 :
                ax = test.plot()
                ax.set_xlabel('time segments')
                ax.set_ylabel('activities')
                fig = ax.get_figure()
                if data_fp == '../A_Sensors.txt':
                    fig.savefig('../images/A/predtarg_ratio/plot_of_%d-th_fold.eps' %i, format='eps')
                else:
                    fig.savefig('../images/B/predtarg_ratio/plot_of_%d-th_fold.eps' % i, format='eps')
            print('Fold: ', i, 'error on test: ', test_err, ' err of ann: ', (1 - nn_accuracy))
            results.append(test_err)
            nn_accuracies.append(nn_accuracy)
        nn_err = np.apply_along_axis(reverse, 0, np.array(nn_accuracies))
        total_errs = pd.DataFrame(np.vstack((np.array(results), nn_err)).T)
        total_errs.columns = ['hybrid model', 'neural network']
        ax = total_errs.plot()
        ax.set_xlabel('k-folds')
        ax.set_ylabel('error')
        if data_fp == '../A_Sensors.txt':
            ax.get_figure().savefig('../images/A/errs.eps', format='eps')
        else:
            ax.get_figure().savefig('../images/B/errs.eps', format='eps')
        print('Average error on tests among k-fold: ', np.mean(np.array(results)))
        print('Average error of nn on tests among k-fold: ', (1 - np.mean(np.array(nn_accuracies))))
        pd.DataFrame(results).to_csv(path_or_buf='../results.csv', index=False, sep=';')
        pd.DataFrame(nn_accuracies).to_csv(path_or_buf='../NNresults.csv', index=False, sep=';')


def execute_my_hmm():
    print('Results on dataset A:')
    cv = CrossValidator('../A_Sensors.txt', '../A_ADLs.txt', 1440, '../dfA.csv', 'A')
    print('Results on dataset B:')
    cv = CrossValidator('../B_Sensors.txt', '../B_ADLs.txt', 1440, '../dfB.csv', 'B')


execute_my_hmm()
