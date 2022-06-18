
import tensorflow as tf

sparse_ae = {

    'batch_size': 32,
    'loss_function': tf.keras.losses.MeanSquaredError(),
    'optimizer': tf.keras.optimizers.Adam(lr=0.001),
    'epochs': 20,

    'encoder': [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            128, activation='relu'
        ),
        tf.keras.layers.Dense(
            64, activation='relu'
        ),
        tf.keras.layers.Dense(
            8, activation='sigmoid',
            activity_regularizer=tf.keras.regularizers.L1(0.001)
        )
    ],

    'decoder': [
        tf.keras.layers.Dense(
            64, activation='relu'
        ),
        tf.keras.layers.Dense(
            128, activation='relu'
        ),
        tf.keras.layers.Dense(
            240, activation='sigmoid'
        )
    ]

}

convolutional_ae = {

    'batch_size': 32,
    'loss_function': tf.keras.losses.MeanSquaredError(),
    'optimizer': tf.keras.optimizers.Adam(lr=0.001),
    'epochs': 20,

    'encoder': [
        tf.keras.layers.Reshape(
            (-1, 1)
        ),
        tf.keras.layers.Conv1D(
            16, kernel_size=3, padding='same', activation='relu'
        ),
        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),
        tf.keras.layers.Conv1D(
            32, kernel_size=3, padding='same', activation='relu'
        ),
        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),
        tf.keras.layers.Conv1D(
            64, kernel_size=3, padding='same', activation='relu'
        ),
        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            8, activation='relu',
            activity_regularizer=tf.keras.regularizers.L1(0.001)
        )
    ],

    'decoder': [
        tf.keras.layers.Dense(
            240 // 2**3 * 64
        ),
        tf.keras.layers.Reshape(
            (240 // 2**3, 64)
        ),
        tf.keras.layers.Conv1DTranspose(
            64, kernel_size=3, padding='same', activation='relu'
        ),
        tf.keras.layers.UpSampling1D(
            size=2
        ),
        tf.keras.layers.Conv1DTranspose(
            32, kernel_size=3, padding='same', activation='relu'
        ),
        tf.keras.layers.UpSampling1D(
            size=2
        ),
        tf.keras.layers.Conv1DTranspose(
            16, kernel_size=3, padding='same', activation='relu'
        ),
        tf.keras.layers.UpSampling1D(
            size=2
        ),
        tf.keras.layers.Conv1D(
            1, kernel_size=3, padding='same', activation='sigmoid'
        ),
        tf.keras.layers.Reshape(
            (240,)
        )
    ]

}
