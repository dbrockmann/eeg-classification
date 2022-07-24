
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, BinaryCrossentropy as BinaryCrossentropyMetric

def build_binary_cf():
    """
    Build binary classifier
    """

    binary_cf = Sequential([
        Dense(
            units=32, 
            activation='relu'
        ),
        Dense(
            units=16, 
            activation='relu'
        ),
        Dense(
            units=1, 
            activation=None
        )
    ])

    binary_cf.compile(
        optimizer=Adam(
            learning_rate=0.0001
        ),
        loss=BinaryCrossentropy(
            from_logits=True
        ),
        metrics=[
            BinaryCrossentropyMetric(name='Loss'),
            BinaryAccuracy(name='Accuracy')
        ]
    )

    return binary_cf
