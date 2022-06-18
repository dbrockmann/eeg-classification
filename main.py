
import tensorflow as tf

from dataset import load_dataset
from preprocessing import prepare_data
from model.autoencoder import Autoencoder
from training.training_autoencoder import train_autoencoder

# load dataset
X, y = load_dataset('./data/')

# apply preprocessing
data, labels = prepare_data(X, y)

# cut last feature
for i, a in enumerate(data):
    data[i] = a[:, :-1]

# extract splitted data
train_data, test_data, val_data = data

# create tensorflow dataset
autoencoder_train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_data))
autoencoder_test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_data))

# create autoencoder model
autoencoder = Autoencoder(train_data.shape[1:], 16)

# train autoencoder model
autoencoder_train_loss, autoencoder_test_loss = train_autoencoder(autoencoder, autoencoder_train_ds, autoencoder_test_ds, show=True)
