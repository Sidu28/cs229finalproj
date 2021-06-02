import os
import csv
import argparse
import numpy as np
from pprint import pprint
import pandas as pd
import random
import matplotlib.pyplot as plt

import csv
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# PE file related imports
import pefile
# import lief

# Relevant modules
import feature_utils
import feature_selector
from utils import *
from progressbar import ProgressBar

numeric_feature_extractors = feature_utils.NUMERIC_FEATURE_EXTRACTORS
alphabetical_feature_extractors = feature_utils.ALPHABETICAL_FEATURE_EXTRACTORS


UNKNOWN = "unk"

def main(good_dir, bad_dir):
    vocab_size = 5000
    embedding_dim = 64
    max_length = 200
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'
    training_portion = .8

    '''
    If a directory is specified, we iterate through it, extracting numerical features
    and saving them to a csv file which is in the 'data' directory
    '''
    alphabetical_feature_extractors = feature_utils.ALPHABETICAL_FEATURE_EXTRACTORS
    good=[]
    c = 0
    for file in (os.listdir(good_dir)):
        #print("File: ", file)
        if not file.startswith('.'):
            file = os.path.join(good_dir, file)
            features, _ = feature_utils.extract_features(file, alphabetical_feature_extractors,numeric=False)
            op_set = set(features['opcode_seq'])
            string = ' '.join(op_set)
            good.append(string)
        c+=1
        if c==10:
            break
    bad = []
    c=0
    for file in os.listdir(bad_dir):
        print("File: ", file)
        if not file.startswith('.'):
            file = os.path.join(bad_dir, file)
            features, _ = feature_utils.extract_features(file, alphabetical_feature_extractors, numeric=False)
            op_set = set(features['opcode_seq'])
            string = ' '.join(op_set)
            bad.append(string)
        c += 1
        if c == 10:
            break


    features = good+bad
    labels = ([0]*len(bad)) + [1]*len(good)

    print(len(features), len(labels))

    train_size = int(len(features) * 0.8)

    train_articles = features[0: train_size]
    train_labels = labels[0: train_size]

    validation_articles = features[train_size:]
    validation_labels = labels[train_size:]

    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_articles)
    word_index = tokenizer.word_index
    dict(list(word_index.items())[0:10])

    train_sequences = tokenizer.texts_to_sequences(train_articles)
    print(train_sequences[10])

    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    print(len(train_sequences[0]))
    print(len(train_padded[0]))

    print(len(train_sequences[1]))
    print(len(train_padded[1]))

    print(len(train_sequences[10]))
    print(len(train_padded[10]))

    validation_sequences = tokenizer.texts_to_sequences(validation_articles)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type,
                                      truncating=trunc_type)

    print(len(validation_sequences))
    print(validation_padded.shape)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
    print(training_label_seq[0])
    print(training_label_seq[1])
    print(training_label_seq[2])
    print(training_label_seq.shape)

    print(validation_label_seq[0])
    print(validation_label_seq[1])
    print(validation_label_seq[2])
    print(validation_label_seq.shape)

    model = tf.keras.Sequential([
        # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # use ReLU in place of tanh function since they are very good alternatives of each other.
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        # Add a Dense layer with 6 units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    num_epochs = 10
    history = model.fit(train_padded, training_label_seq, epochs=num_epochs,
                        validation_data=(validation_padded, validation_label_seq), verbose=2)

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute feature extraction for an input PE file")
    parser.add_argument('--good', type=str, required=False, help="Directory containing PE files to extract features for")
    parser.add_argument('--bad', type=str, required=False, help="Directory containing PE files to extract features for")
    parser.add_argument('--label', type=int, required=False, default=1, help="Label for the PE Files you are processing")
    args = parser.parse_args()

    main(args.good,args.bad)


