# QReg Repository

## Overview

This project implements the Query-Centric Regression, named QReg.
QReg is an ensemble method based on various base regression models.

Current QReg supports linear, polynomial, decision tree, xgboost, gboosting regression as its base models.

## Dependencies
Python 3.6 or higher, requires scipy, xgboost, numpy, scikit-learn

##  How to install
``pip install qregpy``

##  How to use
### Example I
```
from qregpy import qreg
import numpy as np
# The target fitting function is y=x1+2x2
X = np.array([[1,2],[2,5],[3,7],[4,9],[1,3],[2,4], [3,5], [4,2], [5,1]])
y= np.array([5.2, 12, 17.5, 21.2,7.2, 11,13, 7.8, 6.9])

# train the regression
reg = qreg.QReg(base_models=["linear", "polynomial"], verbose=True).fit(X, y)

# make the prediction for point [3,4]
print(reg.predict([[3,4]]))
```
### Example II
```
from qregpy import qreg
import pandas as pd

# load the files
df = pd.read_csv("/data/10k.csv")
headerX = ["ss_list_price", "ss_wholesale_cost"]
headerY = "ss_wholesale_cost"

# prepare X and y
X = df[headerX].values
y = df[headerY].values

# train the regression using base models linear regression and XGBoost regression.
reg = qreg.QReg(base_models=["linear","xgboost"], verbose=True).fit(X, y)

# make predictions
reg.predict([[93.35, 53.04], [60.84, 41.96]])
```
---------------
