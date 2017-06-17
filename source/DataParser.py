import numpy as np
import pandas as pd


def parse_data(data_fp, targets_fp):
    print('Parsing data..', 'Number of parsed activities:')
    ds = pd.read_csv(data_fp, sep='\s+', header=0, parse_dates=[['Start', 'time'], ['End', 'time.1']])
    target_activities = pd.read_csv(targets_fp, sep='\s+',
                                    header=0, parse_dates=[['Start', 'time'], ['End', 'time.1']])
    features = pd.Series(ds.ix[:, 2].unique())
    states = pd.Series(target_activities.ix[:, -1].unique())
    end = target_activities.ix[target_activities.shape[0]-1, 1]
    start = target_activities.ix[0, 0]
    # interval vars
    delta = np.timedelta64(60, 's')
    num_of_time_segments = np.math.ceil((end - start) / delta)
    ann_dataset = np.empty((0, features.shape[0]+1), int)
    ds_counter = 0
    for act in target_activities.iterrows():
        # iterate over t from act's start time to act's end time, stepping with delta
        mapped_act = [states[states == act[1]['Activity']].index[0]]
        t = act[1]['Start_time']
        while t <= act[1]['End_time.1']:
            active_sensors = np.zeros([12])
            temp_count = ds_counter
            while not(ds.ix[temp_count]['Start_time'] >= (t + delta)):
                if max(t, ds.ix[temp_count]['Start_time']) <= min((t + delta), ds.ix[temp_count]['End_time.1']):
                    active_sensors[features[features == ds.ix[temp_count]['Location']].index] = 1
                temp_count += 1
            if ds.ix[ds_counter]['End_time.1'] <= t:
                ds_counter += 1
            new_row = np.concatenate([active_sensors, mapped_act], axis=0)
            ann_dataset = np.append(ann_dataset, [new_row], axis=0)
            t = t + delta
        if 0 == (act[0] % 50):
            print(act[0], '...')
    print('Data parsed.')
    return pd.DataFrame(ann_dataset)
