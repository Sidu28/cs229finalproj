import os
import csv
import argparse
from collections import Counter
import numpy as np
import xgboost as xgb
from time import time
import numpy as np

# PE file related imports
import pefile
# import lief

# Relevant modules
import feature_utils
import feature_selector
from utils import *

UNKNOWN = "unk"

def main(dir1, dir2):

    '''
    If a directory is specified, we iterate through it, extracting numerical features
    and saving them to a csv file which is in the 'data' directory
    '''
    alphabetical_feature_extractors = feature_utils.ALPHABETICAL_FEATURE_EXTRACTORS
    list1=[]
    for file in os.listdir(dir1):
        print("File: ", file)
        if not file.startswith('.'):
            file = os.path.join(dir1, file)
            features, _ = feature_utils.extract_features(file, alphabetical_feature_extractors,numeric=False)
            opcode_list = (features['opcode_ngrams'])
            list1.append(opcode_list)
    list2 = []
    for file in os.listdir(dir2):
        print("File: ", file)
        if not file.startswith('.'):
            file = os.path.join(dir2, file)
            features, _ = feature_utils.extract_features(file, alphabetical_feature_extractors, numeric=False)
            opcode_list = (features['opcode_ngrams'])
            list2.append(opcode_list)

    list = list1 + list2
    ngrams_list = {}
    ngrams_list[5] = list
    features, ngrams_sets, i2ngram, ngram2i = get_all_files_features_and_labels(ngrams_list)

    labels= [1]*(len(os.listdir(dir1))) + [0]*(len(os.listdir(dir2)))

    X, Y = np.array(features), np.array(labels)

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=111)
    X_train, X_test, y_train, y_test = seperate_to_train_and_test(features, labels)
    print("start learning counter...")
    print(learn_with_XGBClassifier(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)))
    X_train, X_test, y_train, y_test = (np.array(X_train) > 0).astype(int), (np.array(X_test) > 0).astype(
            int), np.array(y_train), np.array(y_test),
    print("start learning bool...")
    print(learn_with_XGBClassifier(X_train, y_train, X_test, y_test))

    print("strat learning cv")
    res = learn_with_cv((X > 0).astype(int), Y)
    num_res = "real accuracy:{}".format(list(map(lambda x: (1 - x) * 100, res["accuracy"]["test-merror-mean"])))
    print(res)
    print(num_res)

def seperate_to_train_and_test(features, labels, test_size= 0.2, num_of_classes = 10):
    test_len = int(len(features)*test_size)
    n= int(test_len/num_of_classes)
    train_ftr = list(features)
    train_lbl = list(labels)
    test_ftr = []
    test_lbl = []
    for i in range(0,num_of_classes):
        c = 0
        for x, y in zip(features, labels):
            if y == i:
                test_ftr.append(x)
                test_lbl.append(y)
                c+=1
                if c == n:
                    break
    # delete test from ftr
    for x,y in zip(test_ftr,test_lbl):
        train_ftr.remove(x)
        train_lbl.remove(y)

    # the remaining elements
    sub_len = test_len-len(test_ftr)
    for i in range(0,sub_len):
        temp = random.randint(0, len(train_ftr)-1)
        test_ftr.append(train_ftr[temp])
        test_lbl.append(train_lbl[temp])
        del train_ftr[temp]
        del train_lbl[temp]
    return train_ftr, test_ftr, train_lbl, test_lbl

def learn_with_cv(X,Y):
    t=time()
    early_stopping = 10
    churn_dmatrix = xgb.DMatrix(X,Y)
    params = {"objective": "multi:softmax", "max_depth": 4, "num_class": 10, "silent": 1,
              "seed": 99, }
    cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=6, num_boost_round=30,
                       metrics="merror", as_pandas=True, early_stopping_rounds=early_stopping)
    t= time()-t
    return {"time: ": t, "params: ": params, "early stopping": early_stopping, "accuracy": cv_results}


def learn_with_XGBClassifier(train_data, train_lbl, test_files, test_lbl, lr=0.22,n_esti=40,seed=123):
    train_time = time()
    xg_cl = xgb.XGBClassifier(objective='multi:softmax', num_class= 10, learning_rate=lr,
                                    n_estimators=n_esti, seed=seed)
    xg_cl.fit(train_data, train_lbl)
    train_time = time() - train_time
    test_time = time()
    preds = xg_cl.predict(test_files)
    test_time = time() - test_time
    accuracy = float(np.sum(preds == test_lbl)) / test_lbl.shape[0]
    return {"train time: ": train_time, "test time: ": test_time, "accuracy: ": accuracy*100}

def get_all_files_features_and_labels(ngrams_list, use_unknown=True, unknown_rate=0.01,num_of_files = 1000):

    # get features, labels and other stuff
    ngrams_sets, i2ngram, ngram2i = get_all_possible_ngrams_from_files(ngrams_list, [5], use_unknown, unknown_rate)
    features = make_ngram_counters_for_files(ngrams_list, ngrams_sets, i2ngram, ngram2i, [5], use_unknown)
    return features, ngrams_sets, i2ngram, ngram2i

def get_all_possible_ngrams_from_files(ngrams_lists, ns, use_unknown=True, unknown_precent=0.01):
        """
        Given all the ngrams used in all the training files, finds out which ngrams are used and gives them a unique index.
        :param ngrams_lists: A dict that maps from n to a list,  where the i-th element in the list is a list that contains
            the ngrams (of the specific n) for the i-th file.
        :type ngrams_lists: dict
        :param ns: A list of the "n"s that are used for ngrams.
        :param use_unknown: If True, then an "unknown ngram" token will be added (and given a unique index) for ngrams
            not found in the training files. Also a some of the least common ngrams will be replaced with the
            "unknown ngram" token for training purposes.
        :param unknown_precent: A number between 0 to 1, that indicates what part of all the ngrams is considered not common
        :return: ngrams_sets, i2ngram, ngram2i where:
            * ngrams_sets is a dict that maps from n to a set of all the ngrams used (n-sized)
            * i2ngram is a list of all the ngrams used (without those "unknown" to us) (name means that given an index
                return a ngram)
            * ngram2i is a dict that maps an ngram to its unique index in i2ngram list (given a ngram return index)
        """
        ngrams_lists = {n: [ngram for f in l for ngram in f] for n, l in ngrams_lists.items()}
        ngrams_sets = {n: set(l) for n, l in ngrams_lists.items()}

        if use_unknown:
            counters = Counter([ngram for n in ns for ngram in ngrams_lists[n]])
            not_used = counters.most_common()[:int(-unknown_precent * len(counters)) - 1:-1]
            for ngram in not_used:
                n = len(ngram[0])
                ngrams_sets[n].discard(ngram[0])

        i2ngram = [ngram for n in sorted(list(ngrams_sets.keys())) for ngram in sorted(list(ngrams_sets[n]))]
        if use_unknown:
            i2ngram.append(UNKNOWN)
        ngram2i = {ngram: i for i, ngram in enumerate(i2ngram)}
        return ngrams_sets, i2ngram, ngram2i

def make_ngram_counters_for_files(ngrams_lists, ngram_sets, i2ngram, ngram2i, ns, use_unknown=True):
        """
        For each file, counts how many time a certain ngram is used.
        :param ngrams_lists: A dict that maps from n to a list,  where the i-th element in the list is a list that contains
            the ngrams (of the specific n) for the i-th file.
        :param i2ngram: A list of all the ngrams used (without those "unknown" to us) (name means that given an index
            return a ngram)
        :param ngram2i: A dict that maps an ngram to its unique index in i2ngram list (given a ngram return index)
        :param ngram_sets: A dict that maps from n to a set of all the ngrams used (n-sized)
        :type ngram_sets: dict
        :param ns: A list of the "n"s that are used for ngrams.
        :param use_unknown: If True, then an "unknown ngram" token will be added (and given a unique index) for ngrams
            not found in the training files. Also a some of the least common ngrams will be replaced with the
            "unknown ngram" token for training purposes.
        :return: A list, where element #i is a list of counters that counts how many time an ngram is used (indexes matches
            i2ngram) for i-th file (in ngrams_lists)
        """
        num_files = len(ngrams_lists[ns[0]])
        features = []
        for i in range(num_files):
            file_features = [0] * len(i2ngram)
            for n in ns:
                for ngram in ngrams_lists[n][i]:
                    index = ngram2i[ngram] if not use_unknown or ngram in ngram_sets[n] else ngram2i[UNKNOWN]
                    file_features[index] += 1
            features.append(file_features)
        return features



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute feature extraction for an input PE file")
    parser.add_argument('--dir1', type=str, required=False, help="Directory containing PE files to extract features for")
    parser.add_argument('--dir2', type=str, required=False, help="Directory containing PE files to extract features for")
    parser.add_argument('--label', type=int, required=False, default=1, help="Label for the PE Files you are processing")
    args = parser.parse_args()

    main(args.dir1,args.dir2)

