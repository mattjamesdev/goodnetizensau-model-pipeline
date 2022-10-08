import pickle
from pathlib import Path

import tensorflow as tf

current_dir = Path(__file__).resolve(strict=True).parent
NN_MODELS_DIR = current_dir / Path("models/nn")


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
    # Define the Tokenizer encoder
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size, output_sequence_length=280
    )
    # Unpickle the weights and set them
    with open(NN_MODELS_DIR / Path("Encoder_Weights"), "rb") as fp:
        w = pickle.load(fp)
    encoder.set_weights(w)
    return encoder


# Create the model
def build_nn_model(encoder):
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
                # weights = [emb_matrix],
                # Use masking to handle the variable sequence lengths
                mask_zero=False,
            ),
            tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    # Get the model weights
    model_path = NN_MODELS_DIR / Path("weights")
    model.load_weights(str(model_path))

    # Compile the model
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, label_smoothing=0.01
        ),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), #BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )

    return model
