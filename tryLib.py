#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:16:37 2020

@author: pankaj
"""

import EasyRegression as es
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import  ExtraTreesRegressor
from sklearn.ensemble import   GradientBoostingRegressor
from sklearn.ensemble import  VotingRegressor
from sklearn.ensemble import StackingRegressor
import time
from sklearn.neural_network import MLPRegressor

a = es.EasyRegression()

a.loadFeature('open_smile_num.csv','grp','Open_Smile')
#a.loadFeature('basic_features.csv','ind','Basic_features')
a.loadLabels('JLA_dataset_codes.csv')


groups = pd.read_csv('groups.csv')
a.activateGroups(groups['group-id'])



datasets = pd.read_csv('dataset.csv')
a.activateDatasets(datasets['dataset'])

a.activateLabel('SMU')


start_time = time.time()
a.buildPipeline(['Open_Smile','feature_extraction'])

print("Running time: %s seconds" % (time.time() - start_time))

#x= a.extractFeatures('Open_Smile')
#y = a.Scaling(x,'std')
#z = a.DimRed('pca',y,{'n_components':3,'n_neighbors':2})
#a.regressionModelInitialize()
#a.findParametersAndEvaluate(z,'leave_one_group_out','final',groups['group-id'])
