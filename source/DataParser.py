import numpy as np
import pandas as pd
import math


def parse_data(data_fp, targets_fp):
    ds = pd.read_csv(data_fp, sep='\s+', header=0, parse_dates=[['Start', 'time'], ['End', 'time.1']])
    target_activities = pd.read_csv(targets_fp, sep='\s+',
                                    header=0, parse_dates=[['Start', 'time'], ['End', 'time.1']])
    features = pd.Series(ds.ix[:, 2].unique())
    states = pd.Series(target_activities.ix[:, -1].unique())
    # interval vars
    t = ds.ix[0]['Start_time']  # initialize t to t0
    delta = np.timedelta64(60, 's')
    end = target_activities.ix[target_activities.shape[0]-1, 1]
    start = target_activities.ix[0, 0]
    num_of_time_segments = np.math.ceil((end - start) / delta)
    ann_dataset = np.empty((0,features.shape[0]+1), int)
    ds_counter = 0
    for act in target_activities.iterrows():
        row = pd.Series(np.zeros(features.shape[0]+1))
        # active sensors in the interval activated before the end of the given activity
        features_t = pd.DataFrame(ds.ix[(ds['Start_time'] >= act[1]['Start_time']) & (ds['End_time.1'] <= act[1]['End_time.1'])])
        # iterate over data sensors w.r.t. the considered time segment
        for f in features_t.iterrows():
            # value 1 is assigned to the elem i if the i-th feature has value 1 w.r.t.the considered activity
            feature_id = features[features == f[1]['Location']].index[0]
            row[feature_id] = 1
        row.ix[row.shape[0]-1] = states[states == act[1]['Activity']].index[0]
        # adding many times this row, as many as the number of time segments in the duration of the considered activity
        for i in range(math.ceil((act[1]['End_time.1'] - act[1]['Start_time']) / delta)):
            ann_dataset = np.append(ann_dataset, [row], axis=0)
    return pd.DataFrame(ann_dataset)
