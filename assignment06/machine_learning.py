import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(path, relative_path='./', max_rows=10000):
    columns = ['relevance', 'qid']
    features = ['feature{}'.format(i) for i in range(1, 136 + 1)]
    columns.extend(features)
    feature_df = pd.read_csv(relative_path + path,
                             names=columns,
                             header=None,
                             sep=' ',
                             nrows=max_rows,
                             index_col=False)
    # If you have a malformed file with delimiters at the end of each line,
    # you might consider index_col=False to force pandas to _not_ use the
    # first column as the index (row names)

    # replace useless feature id's in string
    feature_df[features] = feature_df[features].apply(lambda x: x.str.replace('[0-9]+?:', ''))
    feature_df['qid'] = feature_df['qid'].str.replace('qid:', '')
    # convert from object to int or float
    feature_df = feature_df.apply(pd.to_numeric)

    y = feature_df.pop('relevance')  # pop the y column as series
    y = y.where(y == 0, 1)  # Only labels allowed =0 irrelevant and !=0 relevant

    x = feature_df[features]  # don't use qid

    return x, y


def test_parameters():
    train_path = 'MSLR-WEB10K/Fold1/train.txt'
    train_x, train_y = load_data(train_path, relative_path='./', max_rows=10000)
    test_path = 'MSLR-WEB10K/Fold1/test.txt'

    svm = SVC(max_iter=2000)
    params = [
        # linear kernel params
        {
            'kernel': ['linear'],
            'C': [0.5, 0.8, 1.0]
        },
        # rbf kernel
        {
            'kernel': ['rbf'],
            'C': [0.5, 0.8, 1.0],
            'gamma': ['auto', 'scale']
        }
    ]
    scoring = ['f1', 'precision', 'recall']

    cv = GridSearchCV(svm, params, n_jobs=-1, scoring=scoring, cv=2, verbose=2, refit=False)

    cv.fit(train_x, train_y)
    results = pd.DataFrame(cv.cv_results_)
    output = results.set_index(
            ['rank_test_f1', 'rank_test_recall', 'rank_test_precision']
        ).sort_index()[['mean_test_f1', 'mean_test_recall', 'mean_test_precision', 'params']]
    print(output)


def do_sth():
    train_path = 'MSLR-WEB10K/Fold1/train.txt'
    train_x, train_y = load_data(train_path, relative_path='./', max_rows=10000)
    test_path = 'MSLR-WEB10K/Fold1/test.txt'
    test_x, test_y = load_data(test_path, relative_path='./', max_rows=10000)

    svm = SVC(C=0.5, kernel='linear', max_iter=2500)
    svm.fit(train_x, train_y)

    pred_y = svm.predict(test_x)

    report = classification_report(test_y, pred_y,
                                   labels=[0, 1],
                                   target_names=['irrelevant', 'relevant'])
    print(report)


if __name__ == '__main__':
    do_sth()
    # test_parameters()
