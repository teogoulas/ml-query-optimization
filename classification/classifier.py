from statistics import mean

import numpy as np
from numpy import absolute, std
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold, cross_val_score


def svm_classification(X: np.array, y: np.array):

    # define the direct multi output wrapper model
    wrapper = MultiOutputRegressor(SVR(epsilon=0.2))
    # define the evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(wrapper, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force the scores to be positive
    n_scores = absolute(n_scores)
    # summarize performance
    print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

