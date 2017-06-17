import numpy as np
from sklearn import preprocessing


def build_transmat(activity, num_states):
    trans_matrix = np.zeros((num_states, num_states))
    # identifying activity x-th and (x+1)th
    for x in activity.iterrows():
        if x[0] == activity.shape[0] - 2: # the last element has no transition obviously
            break
        a1 = int(x[1].iloc[-1])
        a2 = int(activity.iloc[x[0] + 1].iloc[-1])
        trans_matrix[a1, a2] = trans_matrix[a1, a2] + 1
    # normalizing transition matrix
    return preprocessing.normalize(trans_matrix, norm='l1')
