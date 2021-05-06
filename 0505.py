import sys
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import matplotlib
import xlsxwriter
from datetime import datetime
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

#OLS估计量（模型正确设定）的无偏性、一致性和渐近正态性
#for j in range(0, 1000):
size = 10000
x1 = np.random.exponential(200, size)
x2 = np.random.normal(4, 30, size)
u = np.random.uniform(-1, 1, size)

X = np.column_stack((x1, x2))
X = sm.add_constant(X)

#e = u*(sqrt(x1) + 0.05*x2)
e = zeros(size)
e[0] = u[0]
e[1] = u[1]
for i in range(2,size):
    e[i] = 0.4*e[i-1] - 0.04*e[i-2] + u[i]



beta = np.array([3, 2, 4])
y = np.dot(X, beta) + e

model = sm.OLS(y, X).fit()
print(model.summary())
print(model.params)

print(sm.stats.diagnostic.het_white(model.resid, exog = model.model.exog))

model2 = sm.GLS(y, X).fit()
print(model2.summary())
print(model2.params)

from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(model.resid))

from statsmodels.stats.diagnostic import acorr_ljungbox
print(acorr_ljungbox(model.resid, lags = 3, boxpierce=True))

