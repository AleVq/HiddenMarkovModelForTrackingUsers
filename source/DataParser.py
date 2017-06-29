import numpy as np
import pandas as pd


def parse_data(data_fp, targets_fp):
    print('Parsing data..', 'Number of parsed activities:')
    ds = pd.read_csv(data_fp, sep='\s+', header=0, parse_dates=[['Start', 'time'], ['End', 'time.1']])
    target_activities = pd.read_csv(targets_fp, sep='\s+',
                                    header=0, parse_dates=[['Start', 'time'], ['End', 'time.1']])
    features = ds[['Location', 'Place']].drop_duplicates().reset_index(drop=True)
    states = pd.Series(target_activities.ix[:, -1].unique())  # identify activities
    end = target_activities.ix[target_activities.shape[0]-1, 1]
    start = target_activities.ix[0, 0]
    # interval vars
    delta = np.timedelta64(60, 's')
    # ann_dataset = np.empty((0, features.shape[0]+1), int)
    ann_dataset = []
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
                    active_sensors[features[(features['Location'] == ds.ix[temp_count]['Location']) \
                                        & (features['Place'] == ds.ix[temp_count]['Place'])].index] = 1
                temp_count += 1
                if temp_count >= ds.shape[0]:
                    break
            if ds.ix[ds_counter]['End_time.1'] <= t:
                ds_counter += 1
            new_row = np.concatenate([active_sensors, mapped_act], axis=0)
            # ann_dataset = np.append(ann_dataset, [new_row], axis=0)
            ann_dataset.append(new_row)
            t = t + delta
        if 0 == (act[0] % 50):
            print(act[0], '...')
    # dropping rows with all feature to 0:
    ann_dataset = pd.DataFrame(ann_dataset)
    non_null_features= (ann_dataset.ix[:, :(ann_dataset.shape[1]-2)].T != 0).any()
    ann_dataset = ann_dataset.loc[non_null_features].reset_index(drop=True)
    print('Data parsed.')
    return ann_dataset
