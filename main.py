
from dataset import load_dataset
from preprocessing import split_dataset
from autoencoder import autoencoder_classification
from handcrafted import handcrafted_classification

# load dataset
X, y = load_dataset('./data/')

# split the data in a train, test and validation set in a 70:15:15 ratio
data, labels = split_dataset(X, y)

# number of features to extract
feature_dim = 16

# perform autoencoder feature extraction
# and classification
autoencoder_classification(data, labels, feature_dim=feature_dim)

# perform handcrafted feature extraction
# and classification
handcrafted_classification(data, labels, feature_dim=feature_dim)
