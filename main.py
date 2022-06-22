from dataset import load_dataset
from preprocessing import prepare_data_autoencoder, prepare_data_features

# data preperation
X, y = load_dataset('./data/')
autoencoder_data, autoencoder_labels = prepare_data_autoencoder(X, y)
features_data, features_labels = prepare_data_features(X, y, 16)
