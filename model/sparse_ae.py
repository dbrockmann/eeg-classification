
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import L1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric


def build_sparse_ae(input_dim, latent_dim):
    """
    Build sparse autoencoder

    Args:
        input_dim: input dimensions
        latent_dim: bottleneck dimensions, latent space
    """

    encoder = Sequential([
        Dense(
            units=128, 
            activation='relu'
        ),
        Dense(
            units=64, 
            activation='relu'
        ),
        Dense(
            units=latent_dim, 
            activation='sigmoid',
            activity_regularizer=L1(
                l1=0.005
            )
        )
    ])

    decoder = Sequential([
        Dense(
            units=64, 
            activation='relu'
        ),
        Dense(
            units=128, 
            activation='relu'
        ),
        Dense(
            units=input_dim, 
            activation='sigmoid'
        )
    ])

    sparse_ae = Sequential([encoder, decoder])

    sparse_ae.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss=MeanSquaredError(),
        metrics=[
            MeanSquaredErrorMetric(name='MSE')
        ]
    )

    return sparse_ae