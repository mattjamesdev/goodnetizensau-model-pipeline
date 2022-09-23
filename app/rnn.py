import pickle
from pathlib import Path

import tensorflow as tf


current_dir = Path(__file__).resolve(strict=True).parent
RNN_MODELS_DIR = current_dir / Path("models/rnn")


# Set tensorflow log level
tf.get_logger().setLevel("ERROR")


# Create the encoder and load its weights
def get_encoder(vocab_size):
    """
    Create the encoder and load its weights.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    """
    # Define the TF/IDF encoder
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size, output_sequence_length=280
    )
    # Unpickle the weights and set them
    # with open("models/rnn/Encoder_Weights", "rb") as fp:
    with open(RNN_MODELS_DIR / Path("Encoder_Weights"), "rb") as fp:
        w = pickle.load(fp)
    encoder.set_weights(w)
    return encoder


# Create the model
def build_rnn_model(encoder):
    """
    Build the model and load its weights.

    Parameters
    ----------
    encoder : tf.keras.layers.TextVectorization
        Text vectoriser encoder.

    Returns
    -------
    tf.keras.Sequential
        A trained LSTM model.
    """
    model = tf.keras.Sequential(
        [
            encoder,
            tf.keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()) + 1,
                output_dim=100,
                mask_zero=False,
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.01)
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    # Get the model weights
    model_path = RNN_MODELS_DIR / Path("weights")
    model.load_weights(str(model_path))

    # Compile the model
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, label_smoothing=0.01
        ),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )

    return model
