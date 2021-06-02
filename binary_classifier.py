import numpy as np
import pandas as pd
import argparse
import os

import tensorflow as tf
from tensorflow import keras

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

class BinaryClassifier(object):

    def __init__(self):
        pass

    def train(self, file):
        df = pd.read_csv(file)
        properties = list(df.columns.values)
        properties.remove('label')
        X = df[properties]
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid),
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=1000, batch_size=128)
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print('Test accuracy:', test_acc)


        directory = os.path.join(os.getcwd(), 'models')
        if not os.path.isdir(directory):
            os.mkdir(directory)

        model.save(directory + '/model.baseline')

        print('saved model to disk')

    def predict(self, model, file):

        model = keras.models.load_model(model)
        #X, _ = make_blobs(n_samples=1, centers=2, n_features=1088, random_state=1)
        df = pd.read_csv(file)
        properties = list(df.columns.values)
        properties.remove('label')
        X = df[properties]
        y_test = df['label']

        # make a prediction
        ynew = np.argmax(model.predict(X), axis=-1)
        # show the inputs and predicted outputs
        for i in range(len(ynew)):
            print("X=%s, Truth=%s, Predicted=%s" % (i,y_test[i], ynew[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute feature extraction for an input PE file")
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-predict', action='store_true')
    parser.add_argument('--file', type=str, required=False, help="Input file for training")
    parser.add_argument('--model', type=str, required=False, help="Input file for training")
    parser.add_argument('--test_data', type=str, required=False, help="test file path")


    args = parser.parse_args()

    bc = BinaryClassifier()
    if args.train:
        if args.file:
            bc.train(args.file)
        else:
            parser.error('no file specified')

    if args.predict:
        if os.path.exists(args.model):
            bc.predict(args.model, args.test_data)
        else:
            parser.error('no model found')




