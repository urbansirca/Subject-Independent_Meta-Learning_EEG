
import h5py
with h5py.File('data/preprocessed_KU_eeg/KU_mi_smt.h5', 'r') as f:
    print(list(f.keys()))