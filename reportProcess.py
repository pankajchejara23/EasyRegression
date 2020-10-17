# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


c1v4 = pd.read_csv('coder1final.csv')
c2v4 = pd.read_csv('coder2final.csv')

# function to compute RMSE dimension wise
def getRMSE(c1,c2,dim):
    if dim=="ITO":
        c1['ITO'] = (c1['u1'] + c1['u2'] + c1['u3'] + c1['u4'])/4
        c2['ITO'] = (c2['u1'] + c2['u2'] + c2['u3'] + c2['u4'])/4
        
        
    final1 = c1[dim]
    final2 = c2[dim]
    rmse = mean_squared_error(final1,final2,squared=False)
    
    return rmse


def summary(files,human,label=None):
    data = PrettyTable()


    features = ['Basic','OpenSmile','Basic_Opensmile','Basic_Speechfeature','All']

    strategy = ['train_test','cross_val_mean','leave_group_mean','leave_dataset_mean','stratified_mean']
    data.field_names = ['Features'] + strategy
    
    print('================================================')
    print('             ',label)
    print('------------------------------------------------')
    print('Average Baseline:',human)

    for i in range(len(features)):
        f = features[i]
        fname = files[i] +'.csv'
        #print('Feature:',f)
        row = [None]*len(features)
        df = pd.read_csv(fname)
        
        df.drop(df.tail(1).index,inplace=True)
        for j in range(len(strategy)):
            st = strategy[j]
            #print('  Strategy:',st)
            index = df[[st]].idxmin(axis=0)
            value = "%.2f" % df.loc[index,st].values[0]
            row[j] = df.iloc[index,0].values[0] + '[' +str(value) +']'
    
        data.add_row([f]+row)
    
    print(data)




template = ['easyRegress_report_basic_pca_','easyRegress_report_easyRegress_openSmile_pca_','easyRegress_basicNopenSmile_pca_','easyRegress_BasicPlusSpeechFeatures_','easyRegress_BasicPlusOpenSmilePlusSpeechFeatures_']

dim_files = {}
human_per = {}

for dim in ['ARG','CF','CO','ITO','KE','SMU','STR']:
    dim_files[dim] = [f+dim for f in template]
    human_per[dim] = getRMSE(c1v4,c2v4,dim)
    label = dim + ' Analysis Report'
    summary(dim_files[dim],human_per[dim],label)







files = ['easyRegress_report_basic_feature_pca','easyRegress_report_opensmile_pca','easyRegress_basicNopenSmile_pca_final','easyRegress_BasicPlusSpeechFeatures_final','easyRegress_BasicPlusOpenSmilePlusSpeechFeatures_final']










def plotReport(filename):
    inputf = filename + '.csv'

    r = pd.read_csv(inputf)

    x = np.arange(len(r['model']))
    w=.2

    fig, ax = plt.subplots()
    fig.set_figheight(5.6)
    fig.set_figwidth(7)

    ax.set_xticks(x+w/2)
    ax.set_xticklabels(r['model'],rotation=90)

    
    ax.bar(x,r['train_test'],label='RMSE Training',width=w)
    #ax.bar(x,reg_val_score,label='RMSE Validation',width=w)
    ax.bar(x+w,r['stratified_mean'],label='RMSE Test',width=w)
    ax.set_ylabel('Root Mean Square Error')
    ax.set_xlabel('Regression Models')
    
    ax.hlines(2.06,-0.5,8,color='blue',linestyles='dashed',label='train_test')
    ax.hlines(5.33,-0.5,8,color='green',linestyles='dashed',label='cross_val')
    #ax.hlines(rmse_random,-0.5,8,color='red',linestyles='dashed',label='RMSE random model')
    ax.set_title('RMSE  for various regression models -' + filename)
    ax.legend()
    fig.tight_layout()
    plt.show()