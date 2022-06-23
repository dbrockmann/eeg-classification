
from tensorflow.data import Dataset

from preprocessing import prepare_data_features
from training import train_model
from model.categorical_cf import build_categorical_cf


def handcrafted_classification(X, y, feature_dim=16):
    """
    Performs handcrafted feature extraction with 
    principle component analysis and classification

    Args:
        X: raw data
        y: class labels
        feature_dim: number of features
    """

    # apply preprocessing
    data, labels = prepare_data_features(X, y, feature_dim)

    # extract splitted data
    train_data, test_data, val_data = data
    train_labels, test_labels, val_labels = labels

    # create dataset for classifier
    train_ds = Dataset.from_tensor_slices((train_data, train_labels))
    test_ds = Dataset.from_tensor_slices((test_data, test_labels))

    # shuffle, batch and prefetch
    train_ds = train_ds.shuffle(len(train_data)).batch(32).prefetch(1)
    test_ds = test_ds.shuffle(len(test_data)).batch(32).prefetch(1)

    # build classifier model
    classifier = build_categorical_cf(2)

    # train classifier model
    train_loss, metrics = train_model(
        classifier, train_ds, test_ds, epochs=1000
    )
