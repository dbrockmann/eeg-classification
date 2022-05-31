import os

import numpy as np
import pandas as pd

def load_dataset(data_dir):
    """
    Loads the Bonn dataset (adapted from https://mne.tools/mne-features/auto_examples/plot_seizure_example.html)

    Args:
        data_dir: folder containing the dataset

    Returns:
        data: numpy array of shape(n_samples, n_features) containg the data vectors
        labels: numpy array of shape(n_samples) containing the target values
    """

    data_per_channel = list()
    y = list()
    child_fold = ['F','N','O','S','Z']
    path_child_folds = [os.path.join(data_dir,x) for x in child_fold]

    for path in path_child_folds:
        files = [s for s in os.listdir(path)]
        for file in files:
            # read the da
            _data = pd.read_csv(os.path.join(path,file), engine = 'python', sep = "\r\n", header = None)
            data_per_channel.append(_data.values.T)
        if 'S' in path:
            y.append(np.ones((len(files),)))
        else:
            y.append(np.zeros((len(files),)))

    data = np.concatenate(data_per_channel)
    labels = np.concatenate(y, axis = 0)

    return data, labels
