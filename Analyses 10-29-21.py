import optbinning
from optbinning import BinningProcess
import pandas as pd
from pandas import read_csv
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# changing working directory
os.chdir('U:/VA/ASPIRE/Year 2 Analyses')

# importing data
df = read_csv('FAKEdata.csv')

###################
# Step 1: Binning #
###################

# defining days at home measures variable names
variable_names = list(df.columns[1:5])

# days at home values (in array)
days_at_home = df[variable_names]

# defining PRO outcome measures
PROs = list(df.columns[5:10])

# this defines the binning process
binning_process = BinningProcess(variable_names=variable_names)

binned_df = {} # this is the dictionary that will store the various binned datasets

# this will loop through each predictor (days at home) and each created
# binned variables based on each outcome (PRO), storing the resulting datasets
# (1 for each PRO) in the dictionary 'binned_df'

#for each PRO...
for target in PROs:
    binning_process.fit(df[variable_names], df[target])   

    #for each days-at-home predictor
    for variable in variable_names:
        optb = binning_process.get_binned_variable(name=variable)
        table = optb.binning_table.build()
          
    # adding each table to the dictionary
    binned_df[target] = binning_process.transform(days_at_home, metric="bins")

# this set of loops transforms the bins to numbers for one-hot encoding
OHE_df = {}  #this is the dictionary that will store the one-hot-encoded predictors
for PRO in binned_df:
    base = pd.DataFrame() #defining empty dataframe for storing OHE output
    for variable in variable_names:
        # applying one-hot encoding to the resulting binned variables
        le_binned = 'le_' + variable # 
        le_binned = LabelEncoder()
        encoded = variable + '_encoded'

        binned_df[PRO][encoded] = le_binned.fit_transform(binned_df[PRO][variable])
        
        binned_ohe = variable + '_ohe'
        binned_ohe = OneHotEncoder()
        X = binned_ohe.fit_transform(binned_df[PRO][encoded].values.reshape(-1,1)).toarray()

        
        dfOneHot = pd.DataFrame(X, columns = [variable + str(int(i)) for i in range(X.shape[1])])       
        #THIS IS WHERE I NEED TO REMOVE REDUNDANT COLUMNS AND RESCORE THE DATA!!!
        lastcol = len(dfOneHot.columns)
        dfOneHot = dfOneHot.iloc[:, 1:lastcol]
        
        dfOneHot = dfOneHot[:,:len(dfOneHot.columns)-2]




        
        base = pd.concat([base, dfOneHot], axis=1)

    #adding the one-hot-encoded variables to the datasets in OHE_df
    OHE_df[PRO] = base

# defining the outcomes (PROs) dataset
PROs = df[PROs]

#########################
# Step 2: Modeling PROs #
#########################

from sklearn.datasets import load_boston
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet

# this will loop through dataset in the 'binned_df' dictionary and will
# a) partition the data, 
# b) perform regularized GLM using the days-at-home binned variables via cross-validation, and
# c) score the test set with the final model

for PRO in binned_df:
    
    PRO = 'PRO1'
    
    x, y = OHE_df[PRO].values, PROs[PRO].values

    # split datasets into training and test sets    
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30)
    
    # define the model
    model = ElasticNet()
    # define model evaluation method
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    # define grid for grid search of hyperparameters
    grid = dict()
    grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
    grid['l1_ratio'] = arange(0, 1, 0.01)

    # define search
    search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(xtrain, ytrain)

    # summarize
    print('MAE: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)    

    # next step is to score the test data and calculate MAE







