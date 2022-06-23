
from dataset import load_dataset
from autoencoder import autoencoder_classification
from handcrafted import handcrafted_classification

# load dataset
X, y = load_dataset('./data/')

# number of features to extract
feature_dim = 16

# perform autoencoder feature extraction
# and classification
autoencoder_classification(X, y, feature_dim=feature_dim)

# perform handcrafted feature extraction
# and classification
handcrafted_classification(X, y, feature_dim=feature_dim)
