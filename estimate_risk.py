#!/usr/bin/env python
"""
Predict the risk of borrowers being unable to repay a loan using
logistic regression

Example from: http://datascience-enthusiast.com/R/R_Python_ml.html

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (24.07.2016)"
__email__ = "mdekauwe@gmail.com"

import os
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

def main():

    fname = "loans_imputed.csv"
    df = pd.read_csv(fname)

    X = df[['credit.policy', 'int.rate','installment',\
            'log.annual.inc','dti','fico','days.with.cr.line',\
            'revol.bal','revol.util','inq.last.6mths', 'delinq.2yrs','pub.rec']]
    y = df['not.fully.paid']

    model = LogisticRegression()

    # split 70% for training and 30% for testing
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, random_state=144, test_size=0.3)

    model.fit(X_train, y_train)
    print 'Model accuracy: %f' % (model.score(X_train, y_train) * 100.0)
    #print "Coefficients:", model.coef_
    print "Intercept:",  model.intercept_

    names = list(X.columns.values)

    for i,name in enumerate(names):
        print name, model.coef_[0,i]

    predict = model.predict(X)

    #plt.plot(X, predict, "ro")
    #plt.plot(X, y, "ko")
    #plt.ylim(-0.1, 1.1)
    #plt.show()

if __name__ == "__main__":

    main()
