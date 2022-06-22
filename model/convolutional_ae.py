
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Reshape, Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D
from tensorflow.keras.regularizers import L1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric


def build_convolutional_ae(input_dim, latent_dim):
    """
    Build convolutional autoencoder

    Args:
        input_dim: input dimensions
        latent_dim: bottleneck dimensions, latent space
    """
    
    encoder = Sequential([
        Reshape(
            target_shape=(-1, 1)
        ),
        Conv1D(
            filters=16, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Conv1D(
            filters=32, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Conv1D(
            filters=64, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        ),
        MaxPooling1D(
            pool_size=2
        ),
        Flatten(),
        Dense(
            units=latent_dim, 
            activation='relu',
            activity_regularizer=L1(
                l1=0.001
            ),
            name='bottleneck'
        )
    ])

    decoder = Sequential([
        Dense(
            units=input_dim // 2**3 * 64
        ),
        Reshape(
            target_shape=(input_dim // 2**3, 64)
        ),
        Conv1DTranspose(
            filters=64, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        ),
        UpSampling1D(
            size=2
        ),
        Conv1DTranspose(
            filters=32, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        ),
        UpSampling1D(
            size=2
        ),
        Conv1DTranspose(
            filters=16, 
            kernel_size=3, 
            padding='same', 
            activation='relu'
        ),
        UpSampling1D(
            size=2
        ),
        Conv1D(
            filters=1, 
            kernel_size=3, 
            padding='same', 
            activation='sigmoid'
        )
    ])

    convolutional_ae = Sequential([encoder, decoder])

    convolutional_ae.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss=MeanSquaredError(),
        metrics=[
            MeanSquaredErrorMetric(name='MSE')
        ]
    )

    return convolutional_ae
    