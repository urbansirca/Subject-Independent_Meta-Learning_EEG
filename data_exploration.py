import numpy as np
import h5py
import matplotlib.pyplot as plt

dfile = h5py.File('data/preprocessed_KU_eeg/KU_mi_smt.h5', 'r')

# Get data from single subject.
def get_data(subj):
    dpath = 's' + str(subj)
    X = dfile[dpath]['X']
    Y = dfile[dpath]['Y']
    return X, Y

def get_multi_data(subjs):
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y

# get data from all subjects and inspect dimensions
X, Y = get_multi_data(range(1, 54))
print(X.shape)
print(Y.shape)

# get data from all subjects and inspect dimensions
X, Y = get_multi_data(range(1, 54))
print(X.shape)
print(Y.shape)


# plot first trial (all channels)

plt.figure(figsize=(10, 5))
plt.plot(X[10, :, :].T)
plt.show()