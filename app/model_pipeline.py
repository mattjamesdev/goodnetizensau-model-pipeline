import warnings
import sys
from pathlib import Path

import joblib as jl
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from app.nn import get_encoder, build_nn_model
# from nn import get_encoder, build_nn_model


# Silence scikit-learn UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


# Global variables
VOCAB_SIZE = 5000
BASE_DIR = Path(__file__).resolve(strict=True).parent
CLASSES = ("toxicity", "aggression", "attacking")


# Load the LogisticRegression model, count vectors, and TF/IDF matrix
logistic_model: LogisticRegression = jl.load(f"{BASE_DIR}/models/logistic/logistic.joblib")
count: CountVectorizer = jl.load(f"{BASE_DIR}/models/logistic/count.joblib")
tfidf: TfidfTransformer = jl.load(f"{BASE_DIR}/models/logistic/tdidf.joblib")


# Create the encoder and the RNN model
encoder = get_encoder(VOCAB_SIZE)
nn_model = build_nn_model(encoder)


# Get harsh words using coeffs of the TF/IDF
def get_harsh_words(docs_val, count_vect):
    """
    Get the harsh words from the TF/IDF model.

    Parameters
    ----------
    docs_val : sklearn.feature_extraction.text.TfidfTransformer
        TF/IDF analysis of the sentence.
    count_vect : sklearn.feature_extraction.text.CountVectorizer
        Function that counts the frequency of the terms. The TF part of the TF/IDF
        analysis.

    Returns
    -------
    list[str]
        A list of harsh words.
    """
    docs_val_arr = docs_val.toarray()[0]
    vocabulary = count_vect.get_feature_names_out()
    harsh_words_list = []
    for i in range(len(docs_val_arr)):
        if docs_val_arr[i] > 0.5:
            harsh_words_list.append(vocabulary[i])
    return harsh_words_list


def predict_pipeline(input_text: str) -> tuple:
    """
    Takes in a dict (JSON object) of inputs and returns the response (as an HTTP
    response).

    Parameters
    ----------
    input_text : str
        Text to classify.

    Returns
    -------
    tuple[prediction: int, harsh_words: list[str], probabilities: list[float]]
        Tuple of prediction (cyberbullying or not), list of harsh words, and
        list of probabilities for different categories.
    """

    # Transform the text
    transformed_text = count.transform([input_text])
    transformed_text = tfidf.transform(transformed_text)
    transformed_text = normalize(transformed_text)

    predicted_vector = logistic_model.predict(transformed_text)
    prediction = int(predicted_vector[0])

    # Get harsh words from the LogisticRegression model, and get probabilities from
    # the RNN
    if prediction == 1:
        harsh_words = get_harsh_words(transformed_text, count)
        probabilities_numpy = nn_model.predict([input_text]).tolist()[0]
        probabilities = [round(100 * float(num), 1) for num in probabilities_numpy]
    else:
        harsh_words = []
        probabilities = []

    return prediction, harsh_words, probabilities


if __name__ == "__main__":

    print(BASE_DIR)

    input_text = " ".join(sys.argv[1:])
    print(input_text)

    output = predict_pipeline(input_text)
    print(output)
    for thing in output:
        print(type(thing))
        if type(thing) == list:
            for inner_thing in thing:
                print(inner_thing)
