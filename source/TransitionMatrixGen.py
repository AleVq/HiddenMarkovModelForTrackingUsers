import numpy as np
import pandas as pd

class HMM1:
    def __init__(self, data_fp, targets_fp):
        self.ds = pd.read_csv(data_fp, sep='\s+', header=0, parse_dates=[['Start', 'time'], ['End', 'time.1']])
        self.target_activities = pd.read_csv(targets_fp, sep='\s+',
                                             header=0, parse_dates=[['Start', 'time'], ['End', 'time.1']])
        self.n_activity = pd.Series(self.target_activities.ix[:,2].unique())


    def cons_tab_trans(self, activity, target):
        self.n_activity = activity
        self.target_activities = target
        tab_trans=np.zeros((self.n_activity.size,self.n_activity.size))
        # Doppio for per identificare le attività y successive all'attività x in target_activities
        for x in self.n_activity.values:
            for y in range(0,(self.target_activities.values[:,2].size)-1):
                if x == self.target_activities.values[y,2]:
                    # Recupero gli indici numeri delle attività per aggiornare la tabella delle transizioni
                    a1 = self.n_activity[self.n_activity == x].index[0]
                    a2 = self.n_activity[self.n_activity == self.target_activities.values[y+1,2]].index[0]
                    tab_trans[a1,a2] = tab_trans[a1,a2]+1
        # Normalizzo la tabella delle transizioni
        sum_rows = tab_trans.sum(axis=1)
        for x in range(0, tab_trans[0,:].size):
            for y in range(0,tab_trans[:,0].size):
                tab_trans[x,y] = tab_trans[x,y]/sum_rows[x]
        print(pd.DataFrame(tab_trans))

def execute_my_hmm():
    my_hmm = HMM1('../A_Sensors.txt', '../A_ADLs.txt')
    my_hmm.cons_tab_trans(my_hmm.n_activity,my_hmm.target_activities)

execute_my_hmm()
