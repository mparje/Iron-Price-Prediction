#!/usr/bin/env python
# coding: utf-8

# In[1]:


# here we will import the libraries used for machine learning
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O, data manipulation
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from pandas import set_option
plt.style.use('ggplot') # nice plots

from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import RandomizedSearchCV  # Randomized search on hyper parameters.
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import GridSearchCV
import imgkit

import warnings
warnings.filterwarnings('always')

def process_data(input_file, delete_nan=False, train=False):

    data = pd.read_csv(input_file, skipinitialspace=True)
    instance_id = data['ID']
    data.drop('ID', axis=1, inplace =True) # drop column "name"

    return data, instance_id


#train data
data, i_id = process_data('dataset_voltage/train.csv', delete_nan=False, train=True)

y = data['V(out)']     # target classes
features = data.drop('V(out)', axis=1, inplace=False)

#test data
data_test, instance_id = process_data('dataset_voltage/test.csv')


set_option('display.width', 100)
set_option('precision', 2)

print("SUMMARY STATISTICS OF NUMERIC COLUMNS")

#print(data.describe().T)
data.sample(10)

print("Correlation matrix")

corr = data.corr()
df_styled = corr.style.background_gradient(cmap='coolwarm')

import dataframe_image as dfi
dfi.export(df_styled, 'Correlation Matrix.png')

# <a id='ml'></a>
# ## Machine Learning: Regression models
#
#
# To build machine learning models the original data was divided into features (X) and target (y) and then split into train (80%) and test (20%) sets. Thus, the algorithms would be trained on one set of data and tested out on a completely different set of data (not seen before by the algorithm).
#
# <a id='sp'></a>
# ### Splitting the data into train and test sets

# In[54]:


# Original dataset
#oversample = SMOTE(sampling_strategy=0.27)

'''
oversample = SMOTE(sampling_strategy='minority')

under = RandomUnderSampler()
'''

y = data['V(out)']
X = data.drop('V(out)', axis=1)

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, stratify=y, random_state=42)



#X, y = under.fit_resample(X, y)

X.info()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4)

X2, y2 = X, y
#X_train, y_train = oversample.fit(X_train, y_train)

print("Output voltage: ")


# Create the random grid
param_dist = {'n_estimators': [5, 10, 20, 25, 50],
               "max_features": [2,3,4,5,6,7,8,9],
               'max_depth': [2,3,4,5,6,7,8,9,10,12],
              }

rf = RandomForestRegressor()

#rf = xgboost.XGBRFRegressor()

rf_cv = RandomizedSearchCV(rf, param_distributions=param_dist, #cv=3,
                            random_state=0, n_jobs=-1)

rf_cv.fit(X, y)

print("Tuned Random Forest Parameters: %s" % (rf_cv.best_params_))

Ran = RandomForestRegressor(n_estimators=50, max_features=7, max_depth=9, random_state=0)

Ran_2 = RandomForestRegressor(n_estimators=50, max_features=7, max_depth=9, random_state=0)
'''
Ran = xgboost.XGBRFRegressor(reg_lambda=0.26, n_estimators=75, max_depth=9, random_state=0)

Ran_2 = xgboost.XGBRFRegressor(reg_lambda=0.26, n_estimators=75, max_depth=9, random_state=0)
'''
Ran.fit(X_train, y_train)
Ran_2.fit(X2, y2)

main_prediction = Ran_2.predict(data_test)

data_predicted = pd.DataFrame()

data_predicted['ID'] = instance_id
data_predicted['V(out)'] = main_prediction

#print("TOTAL NUMBER OF Defaults: ", main_prediction.astype(bool).sum())


import time
timestr = time.strftime("%Y%m%d-%H%M%S")

data_predicted.to_csv('Predictions_voltage/output_voltage'+timestr+'.csv', index=False)

y_pred = Ran.predict(X_test)

errors = abs(y_pred - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Random forest Accuracy:', accuracy)
