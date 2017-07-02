import numpy as np
import pandas as pd
from HybridHMM import HybridHMM
from DataParser import parse_data
import os.path
import math
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


# input: data and targets file-paths, k of k-fold,
# input: length of observations' sequence for viterbi algorithm
# input: size of test set for the k-fold cross val.
def cross_validation(data_fp, targets_fp, subset_size, fp, ds_label):
    # parse data only if the aligned dataset has not been generated yet
    if not(os.path.isfile(fp)):
        temp, states_labels, features_labels = parse_data(data_fp, targets_fp)
        temp.to_csv(path_or_buf=fp, index=False, sep=';')
        pd.DataFrame(states_labels).to_csv(path_or_buf='../csvs/labels_states_%s.csv' %ds_label, sep=',', header=False)
        pd.DataFrame(features_labels).to_csv(path_or_buf='../csvs/labels_features_%s.csv' %ds_label, sep=',', header=False)
    ds = pd.read_csv(fp, header=0, sep=';')
    k = math.floor(len(ds) / subset_size)
    hybrid_errors = []  # contains the percentage of non-matched prediction for each fold
    nn_accuracies = []
    predictions = np.empty((0))
    targets = np.empty((0))
    # splitting dataset
    for i in range(k):
        test_set = ds.ix[i * subset_size:][:subset_size].reset_index(drop=True)
        if i == 0:
            training_set = ds.ix[(i + 2) * subset_size:].reset_index(drop=True)
        else:
            training_set = pd.concat([ds.ix[:i * subset_size],
                                      ds.ix[(i + 2) * subset_size:]]).reset_index(drop=True)
        # hybrid model's initialization, training and testing
        model = HybridHMM(training_set)
        model.train(training_set)
        test_err, nn_accuracy, prediction, target = model.test(test_set)
        predictions = np.append(predictions, prediction)
        targets = np.append(targets, target)
        test = pd.DataFrame(np.vstack((prediction, target)).T)  # getting prediction and targets together
        # printing just a couple of diagrams of predictions vs targets trends
        if i < 2:
            ax = test.plot()
            ax.set_xlabel('time segments')
            ax.set_ylabel('activities')
            fig = ax.get_figure()
            if data_fp == '../A_Sensors.txt':
                fig.savefig('../images/A/predtarg_ratio/plot_of_%d-th_fold.eps' %i, format='eps')
            else:
                fig.savefig('../images/B/predtarg_ratio/plot_of_%d-th_fold.eps' % i, format='eps')
        print('Fold: ', i, 'error on test: ', test_err, ' err of ann: ', (1 - nn_accuracy))
        hybrid_errors.append(test_err)
        nn_accuracies.append(nn_accuracy)
    # plotting neural network's and hybrid's errors
    nn_err = np.apply_along_axis(lambda x: 1-x, 0, np.array(nn_accuracies))  # neural network's error
    compared_errs = pd.DataFrame(np.vstack((np.array(hybrid_errors), nn_err)).T)
    compared_errs.columns = ['hybrid model', 'neural network']
    ax = compared_errs.plot()
    ax.set_xlabel('k-folds')
    ax.set_ylabel('error')
    if ds_label == 'A':
        ax.get_figure().savefig('../images/A/errs.eps', format='eps')
    else:
        ax.get_figure().savefig('../images/B/errs.eps', format='eps')
    print('Average error on tests among k-fold: ', np.mean(np.array(hybrid_errors)))
    print('Average error of nn on tests among k-fold: ', (1 - np.mean(np.array(nn_accuracies))))
    mean_errs = [ds_label, np.mean(np.array(hybrid_errors)), (1 - np.mean(np.array(nn_accuracies)))]
    return predictions, targets, mean_errs


def evaluate(predictions, targets, ds_label):
    plotting_data = pd.DataFrame(np.vstack((predictions, targets)).T)  # getting prediction and targets together
    ax = plotting_data.plot()
    ax.set_xlabel('time segments')
    ax.set_ylabel('activities')
    fig = ax.get_figure()
    if ds_label == 'A':
        fig.savefig('../images/A/predtarg_ratio/total_ratio.eps', format='eps')
    else:
        fig.savefig('../images/B/predtarg_ratio/total_ratio.eps', format='eps')
    confusion_matrix_sklearn = metrics.confusion_matrix(y_pred=predictions, y_true=targets)
    pd.DataFrame(confusion_matrix_sklearn).reset_index(drop=True).to_csv(path_or_buf='../csvs/conf_matrix%s.csv' %ds_label, sep=',')
    tot_accuracy = metrics.accuracy_score(y_true=targets, y_pred=predictions)
    tot_precision = metrics.precision_score(y_true=targets, y_pred=predictions, average='weighted')
    f_measure = metrics.f1_score(y_true=targets, y_pred=predictions, average='weighted')
    tot_recall = metrics.recall_score(y_true=targets, y_pred=predictions, average='weighted')
    evaluation = pd.DataFrame([[ds_label, tot_accuracy, tot_precision, tot_recall, f_measure]])
    evaluation.columns = ['dataset', 'accuracy', 'precision', 'recall', 'f measure']
    return evaluation


def execute_my_hmm():
    print('Results on dataset A:')
    predictions1, targets1, mean_errs1 = cross_validation('../A_Sensors.txt', '../A_ADLs.txt', 1440, '../dfA.csv', 'A')
    ev1 = np.around(evaluate(predictions1, targets1, 'A'), decimals=5)
    print('Results on dataset B:')
    predictions2, targets2, mean_errs2 = cross_validation('../B_Sensors.txt', '../B_ADLs.txt', 1440, '../dfB.csv', 'B')
    total_mean_errors = pd.DataFrame(np.vstack((mean_errs1, mean_errs2)))
    total_mean_errors.columns = ['dataset', 'mean hybrid error', 'mean neural network error']
    total_mean_errors.to_csv(path_or_buf='../csvs/total_mean_errors.csv', sep=',', index=False)
    ev2 = np.around(evaluate(predictions2, targets2, 'B'), decimals=5)
    pd.concat([ev1, ev2]).reset_index(drop=True).to_csv(path_or_buf='../csvs/final_results.csv', sep=',', index=False)

execute_my_hmm()
