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

    data_per_recording = list()
    labels_per_set = list()
    child_fold = ['F','N','O','S','Z']
    path_child_folds = [os.path.join(data_dir,x) for x in child_fold]

    for path in path_child_folds:
        files = [s for s in os.listdir(path)]
        for file in files:
            # read the data from the text files
            _data = pd.read_csv(os.path.join(path,file), engine = 'python', sep = "\r\n", header = None)
            data_per_recording.append(_data.values.T)
        # label eeg recordings that contain seizures with 1, all others with 0 (binary classification)
        if 'S' in path:
            labels_per_set.append(np.ones((len(files),)))
        else:
            labels_per_set.append(np.zeros((len(files),)))

    data = np.concatenate(data_per_recording)
    labels = np.concatenate(labels_per_set, axis = 0)

    return data, labels
