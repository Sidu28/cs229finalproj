import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
import utils


'''
Uses sklearn SelectKBest to extract features
Uses Chi squared statistical measure (often used for classification)
Reduces feature dimension to 'num_features'
Reads from input_file, writes to output_file
'''
def select_features(num_features, input_file, output_file=None, num_print=10):
    # Data from csv
    df = pd.read_csv(input_file).fillna(0)
    X = df.drop('label', axis=1)
    y = df['label']
    print(output_file)
    # Feature selection model
    fs = SelectKBest(score_func=chi2, k=num_features)
    model = fs.fit(X, y)

    # Print best columns and scores
    if num_print:
        dfscores = pd.DataFrame(model.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        feature_scores = pd.concat([dfcolumns,dfscores],axis=1)
        feature_scores.columns = ['Feature Name','Score']
        print(feature_scores.nlargest(num_print,'Score')) 

    # Select best columns and optionally save
    cols = fs.get_support(indices=True)
    df_selected = df.iloc[:,cols]
    df_selected = df_selected.assign(label=y.values)
    if output_file:
        print("Hello")
        df_selected.to_csv(output_file)

'''
For given feature name, draw sample (if sample_size is specified) 
from good and malicious files and plot their distributions.
'''
def compare_feature(feature_name, data_file, output_file=None, sample_size=None):
    # Data from csv
    df = pd.read_csv(data_file)
    df = df[[feature_name, 'label']]
    df_pivot = df.pivot(columns='label', values=feature_name)

    df_plot = df_pivot.rename(columns={0: 'Malicious', 1: 'Benign'})

    # Sample
    if sample_size:
        good_sample = df_plot['Benign'].sample(sample_size)
        bad_sample = df_plot['Malicious'].sample(sample_size)
        df_plot = pd.concat([good_sample, bad_sample], axis=1)

    # Plot
    df_plot.plot.density()
    plt.xlabel("Feature Value")
    if output_file:
        plt.savefig(output_file)
    plt.show()
