import os
import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import seaborn as sns


# read input data from program or from csv file
def load_data(path):
    df = pd.read_csv(path)
    return df

def train_and_predict():
    df = load_data(sys.argv[1])
    properties = list(df.columns.values)
    properties.remove('label')
    X = df[properties]
    y = df['label']
    print(X.shape)
    print(y.shape)
    print(X.shape)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


    # train and predict
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))

    # model quality
    print("Training set score: {:.3f}".format(model.score(x_train,y_train)))
    print("Testing set score: {:.3f}".format(model.score(x_test,y_test)))


    def lr_model(x):
        return 1 / (1 + np.exp(-x))
    # plot the loss of the test data
    loss = lr_model(x_test * model.coef_ + model.intercept_).values.ravel()
    plt.plot(x_test, loss, color='black', linewidth=3)



if __name__=="__main__":
    train_and_predict()