import numpy as np
from sklearn import preprocessing


def randomized_mat(matrix):
    rand_mat = np.random.random((matrix.shape[0], matrix.shape[1]))
    return preprocessing.normalize(rand_mat, norm='l1')


class HMM:
    # pi: states priors, t: transition matrix, o: observations emissions matrix
    # all inputs are supposed to be np.arrays
    def __init__(self, pi, t, o):
        self.pi = pi
        self.t = t
        self.o = o

    # o_space: observations' space, s_space: states' space, seq: obs. sequence
    # all input must be np.array
    # output most likely sequence
    def viterbi(self, o_space, s_space, seq):
        T1 = np.zeros((s_space.shape[0], seq.shape[0]))
        T2 = np.zeros((s_space.shape[0], seq.shape[0]))
        for s in range(s_space.shape[0]):
            # TODO updated_o = bayes_rule(self.ann.get_prediction(seq[1]), self.t_priors, f_priors)
            updated_o = randomized_mat(self.o)
            T1[s, 1] = self.pi[s] * updated_o[s, seq[0]]
            T2[s, 1] = 0
        for obs in range(seq.shape[0]):
            for s in range(s_space.shape[0]):
                # TODO updated_o = bayes_rule(self.ann.get_prediction(seq[obs]), self.t_priors, f_priors)
                updated_o = randomized_mat(self.o)
                pw_multiplied = np.multiply(T1[:, obs-1], self.t[:, s]) # point-wise multiply
                max = np.argmax(pw_multiplied)
                T1[s, obs] = updated_o[s, obs] * max
                T2[s, obs] = max
        z = np.zeros((T1.shape[1]))
        for t in range(T1.shape[1]):
            z[t] = np.argmax(T1[:, t])
        result = np.zeros((z.shape[1]))
        for t_z in range(z.shape[0]):
            result = z[t_z]  # z contains states indices
        for i in range(z.shape[0]-1, 1, 1):
            z[i-1] = T2[z[i], i]
            result[i-1] = s[z[i-1]]
        return result


pi = np.array([0.7, 0.3])
t = np.array([[0.6, 0.4], [0.8, 0.2]])
o = np.array([[0.4, 0.1, 0.5], [0.1, 0.15, 0.75]])
hmm = HMM(pi, t, o)
