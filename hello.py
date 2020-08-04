from flask import Flask, render_template, request, make_response, jsonify

import numpy as np
import logging

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow import keras

import pickle

app = Flask(__name__)


def read_or_new_pickle_tokenizer(path):
    try:
        with open(path, "rb") as f:
            bert_tokenizer = pickle.load(f)
            logging.info("Opened the pickle file")
    except Exception:
        bert_tokenizer = load_bert_tokenizer()
        logging.info("Exception opening the pickle file")
        with open(path, "wb") as f:
            pickle.dump(bert_tokenizer, f)
            logging.info("Dumped the pickle file")

    return bert_tokenizer


def load_bert_tokenizer():
    import tensorflow_hub as hub
    import bert

    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    return tokenizer


def tokenize_sentences(sentence):
    return read_or_new_pickle_tokenizer('bert_tokenizer_pickle').\
        convert_tokens_to_ids(read_or_new_pickle_tokenizer('bert_tokenizer_pickle')
                              .tokenize(sentence))


def prediction(sentence):
    txt = sentence
    max_length = 50
    seq = [tokenize_sentences(txt)]
    reconstructed_model = keras.models.load_model("text_model_local")
    padded = pad_sequences(seq, maxlen=max_length)
    pred = reconstructed_model.predict(padded)

    data = {'joy': 0, 'fear': 1, 'anger': 2, 'sadness': 3, 'disgust': 4, 'shame': 5, 'guilt': 6, 'empty': 7,
            'enthusiasm': 8, 'neutral': 9, 'worry': 10, 'surprise': 11, 'love': 12, 'fun': 13, 'hate': 14,
            'happiness': 15,
            'boredom': 16, 'relief': 17}

    new_data = {j: i for i, j in data.items()}

    logging.info(pred, new_data[int(pred.argmax(1))])

    return new_data[int(pred.argmax(1))]


@app.route('/predict-emotion', methods=['POST'])
def do_prediction():
    # Validate the request body contains JSON
    if request.is_json:

        # Parse the JSON into a Python dictionary
        req = request.get_json()

        response_body = {
            "sentence": req.get("sentence"),
            "prediction": prediction(req.get("sentence"))
        }

        res = make_response(jsonify(response_body), 200)

        return res

    else:

        # The request body wasn't JSON so return a 400 HTTP status code
        return "Request was not JSON", 400
