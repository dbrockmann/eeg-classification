
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
