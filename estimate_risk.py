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
from patsy import dmatrices

def main():

    fname = "loans_imputed.csv"
    df = pd.read_csv(fname)

    #print df.describe()
    df.hist()
    plt.show()

    # clean up the dataframe
    df.rename(columns={'not.fully.paid': 'not_fully_paid',
                       'credit.policy': 'credit_policy',
                       'int.rate': 'int_rate',
                       'log.annual.inc': 'log_annual_inc',
                       'days.with.cr.line': 'days_with_cr_line',
                       'revol.bal': 'revol_bal',
                       'inq.last.6mths': 'inq_last_6mths',
                       'delinq.2yrs': 'delinq_2yrs',
                       'pub.rec': 'pub_rec'}, inplace=True)

    y, X = dmatrices('not_fully_paid ~ credit_policy + int_rate + \
                     installment + log_annual_inc + dti + \
                     days_with_cr_line + revol_bal + inq_last_6mths + \
                     delinq_2yrs + pub_rec',
                     df, return_type='dataframe')

    model = LogisticRegression()
    model.fit(X, y)
    predict = model.predict(X)

    print
    print
    print 'Model accuracy: %f' % (model.score(X, y) * 100.0)
    print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
    


    #plt.plot(X, predict, "ro")
    #plt.plot(X, y, "ko")
    #plt.ylim(-0.1, 1.1)
    #plt.show()

if __name__ == "__main__":

    main()
