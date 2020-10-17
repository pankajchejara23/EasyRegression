
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import seaborn as sns
from sklearn.model_selection import train_test_split
import math
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error
import traceback
import statistics
# Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import RANSACRegressor
from pyfiglet import Figlet
from sklearn.model_selection import cross_val_score
from joblib import dump, load


from sklearn.kernel_ridge import KernelRidge
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

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import cross_val_score


import statistics
from sklearn.model_selection import cross_validate

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn import manifold

import numpy as np
from sklearn.model_selection import GridSearchCV
from scipy.special import entr

import random
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics


from art import *




class EasyRegression:
    def __init__(self):
        print(text2art('Easy'))
        print(text2art('Regression'))

        self.seed = 40
        self.strategy = None
        
        self.parameterFound = dict()
        
        self.configured = False
        
        self.models = None
    
        # Scalers
        self.std = StandardScaler()
        self.mmax = MinMaxScaler()
        
        self.random_state = 42
        self.feature_set = dict()
        self.label_set = dict()
        
        self.groups = None
        self.datasets = None
        self.label = None

        self.flagDataset = False
        self.flagGroup = False

        self.flagParameterFind = False
        
        
        self.train_test = None
        self.cross_val = None
        self.leave_group = None
        self.leave_dataset = None
        self.stratified = None

    def loadFeature(self,feature_file,feature_type,feature_name):
        
        if len(self.feature_set) == 0:
            print('-------------------------------')
            print(' STEP : Features loading')
            print('-------------------------------')
        if feature_type not in ['ind','grp']:
            print('===> Error: Undefined feature type')
            return 
        else:
            try:
                
                if feature_name in self.feature_set.keys():
                    print('===> Feature with name ',feature_name,' already exist. Choose a different name')
                    return 
                else:
                    tmp = pd.read_csv(feature_file)
                    
                    if len(self.feature_set) > 0:
                        first_feat = self.feature_set[list(self.feature_set.keys())[0]]
                        if tmp.shape[0] != first_feat[2].shape[0]:
                            print('===> Error: Mismatch in feature size with previously added features ',first_feat[1] )
                            return 
                    
                    self.feature_set[feature_name] = [feature_type,feature_name,tmp]
                    print('===> Feature file:',feature_file,' is loaded successfully !')
                    print('===> Summary:')
                    print('     #instances:',tmp.shape[0])
                    print('     #attributes:',tmp.shape[1])
                    num_cols = tmp.select_dtypes(['int64','float64'])
                    print('     #numeric-attributes:',num_cols.shape[1])
                    print('')
                    return num_cols
            except:
                print('===> Error occurred while loading the file')
                traceback.print_exc()
                
                
    def loadLabels(self,label_file):
        try:
            print('-------------------------------')
            print(' STEP : Labels loading')
            print('-------------------------------')
            tmp = pd.read_csv(label_file)
            
            if len(self.feature_set) > 0:
                first_feat = self.feature_set[list(self.feature_set.keys())[0]]
                if tmp.shape[0] != first_feat[2].shape[0]:
                    print('  Error: Mismatch in feature size with loaded feature ',first_feat[1] )
                    return None
                
            for label in tmp.columns:
                self.label_set[label] = tmp[label]
            print('===> Label file:',label_file,' is loaded successfully !')
            print('===> Summary:')
            print('     #labels:',len(tmp.columns.tolist()))
            print('     labels:', tmp.columns.tolist())
            print('')
            return tmp
        except:
            print('===> Error occurred while loading the file:',label_file)
            traceback.print_exc()
            return None
    


                
    def feature_name_check(self,feature_name):
        if feature_name not in self.feature_set.keys():
            print(' Feature name:', feature_name,' is not available.')
            return None
    
    def label_name_check(self,label_name):
        if label_name not in self.label_set.keys():
            print(' Label name:',label_name,' is not available.')
            return None
            
    
    def extractFeatures(self,data,cor=.80):
        print('-------------------------------')
        print(' STEP : Feature Extraction     ')
        print('-------------------------------')
        
        
        
        
        correlated_features = set()
        features = data
        correlation_matrix = features.corr()
        for i in range(len(correlation_matrix .columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > cor:
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)
        #print('Correlated Features:')
        #print(correlated_features)
        
        features.drop(labels=correlated_features,axis=1,inplace=True)
        print('===> ',len(correlated_features),' correlated features are removed.')
        print('===>  Final features shape:',features.shape)
        
        return features
    


    def findCorrelation(self,label_name=None,sort=True):
        if self.dataReady == False:
            print('Data is not ready yet for analysis.')
            return
        if label_name is not None:
            if label_name in self.labels.columns:
                tmp_features = self.features.copy()
                tmp_features[label_name] = self.labels[label_name]
                cor_table = tmp_features.corr()
                print('           Correlation ')
                print(' -------------------------------')
                print(cor_table[label_name])
                print(' -------------------------------')
        else:    
            if self.labels.shape[1] > 1:
                print(' There are more than one label available.')
                print(self.labels.columns)
                print('Deafult: first column is used to computer correlation')
                label_name = self.labels.columns[0]
                tmp_features = self.features.copy()
                tmp_features[label_name] = self.labels[label_name]
                cor_table = tmp_features.corr()
                print('           Correlation ')
                print(' -------------------------------')
                print(cor_table[label_name])
                print(' -------------------------------')
    
    
    def setGroupFeatureLabels(self,feat_labels):
        self.group_feature_labels = feat_labels
    
    
    """
        This function performs group-level feature computation
        supported fusions: Dimensionality reduction, Entropy, Gini, Average
    """
    def getGroupFeatures(self,data):
        
        
        group_feature_labels = ['add','del','speak','turns']
        
        features_group = dict()
        # iterate for each group-level feature 
        for grp_feature in group_feature_labels:
            
            tmp = list()
            
            # get all column names similar to grp_feature
            for indiv_feature in data.columns:
                if grp_feature in indiv_feature:
                    tmp.append(indiv_feature)
            
            features_group[grp_feature] = tmp.copy()
        
        return features_group
        
    
    # preparing gini coefficient
    def getGINI(self,data):
        """Calculate the Gini coefficient of a numpy array."""
        
        print('-------------------------------')
        print(' STEP : Feature Fusion using Gini')
        print('-------------------------------')
        group_features = self.getGroupFeatures(data)
        
        
        gini = dict()
        for key in group_features.keys():
            
            
            tmp = data[group_features[key]].values
            tmp = tmp + 0.0000001 
            tmp = np.sort(tmp)
            index =  np.arange(1,tmp.shape[1]+1) 
            n = tmp.shape[1]
            

            
            key = 'grp_gini_'+key
            gini[key] = ((np.sum((2 * index - n  - 1) * tmp,axis=1)) / (n * np.sum(tmp,axis=1))) #Gini coefficient
        
        gini_features = pd.DataFrame(gini)
        return gini_features
    

    

    # Compute entropy features for individual features
    def getEntropy(self,data):
        print('-------------------------------')
        print(' STEP : Feature Fusion using Entropy')
        print('-------------------------------')
        group_features = self.getGroupFeatures(data)
        entropy = dict()
        for key in group_features.keys():      
            tmp = data[group_features[key]].values
            tmp = tmp
            tmp_sum = tmp.sum(axis=1,keepdims=True) + .0000000000001
            p = tmp/tmp_sum
            
            key = 'grp_entropy_'+key
            entropy[key] = entr(p).sum(axis=1)/np.log(2)
  
        entropy_features = pd.DataFrame(entropy)
        return entropy_features
    
    
    """
    Apply dimentionality reduction on features
    PCA
    
    """
    def Scaling(self,data,algo):
        
        
        print('-------------------------------')
        print(' STEP : Feature Scaling')
        print('-------------------------------')
        
        
        if algo in ['std','mmax']:
            if algo == 'std':
                res = pd.DataFrame(self.std.fit_transform(data), columns=data.columns)
                print('===> Successfully applied Standard Scaling')
                return res
            elif algo == 'mmax':
                res =  pd.DataFrame(self.mmax.fit_transform(data), columns=data.columns)
                print('===> Successfully applied MinMax Scaling')
                return res
        else:
            print('===> Error: Unsupported scaling method')
            return None
            
    def DimRed(self,algo,data,params=None):
        print('-------------------------------')
        print(' STEP : Feature fusion using DimRed')
        print('-------------------------------')
        if algo not in ['pca','mds','isomap','tsne']:
            print('===> Erro: Unsupported dimension reduction algorithm specified')
            return None
        else:
            if algo!='pca' and len(params) ==0:
                print('===> Error: Specify n_components/n_neighbors parameters')
                return None
            else:    
                # Dimensionality reduction
    
                X_train, X_test, y_train, y_test = train_test_split(data,self.label_set[self.label],train_size=.7,random_state=self.seed)
                
                self.pca = PCA(random_state = self.seed)
                self.mds = manifold.MDS(n_components=params['n_components'],max_iter=100,n_init=1,random_state = self.seed)
                self.isomap = manifold.Isomap(n_neighbors=params['n_neighbors'],n_components=params['n_components'])
                self.tsne = manifold.TSNE(n_components=params['n_components'],init='pca',random_state = self.seed)
            
                if algo == 'pca':
                    self.pca.fit(X_train)
                    pca_features = self.pca.transform(data)
                    print('===> Successfully applied PCA')
                    pca_columns = [None] * pca_features.shape[1]
                    for k in range(pca_features.shape[1]):
                        pca_columns[k] = 'pca_' + str(k)
                    
                    return pd.DataFrame(pca_features,columns=pca_columns) 
                if algo == 'mds':
                    self.mds.fit(X_train)
                    mds_features = self.mds.transform(data)
                    mds_columns = [None] * mds_features.shape[1]
                    for k in range(mds_features.shape[1]):
                        mds_columns[k] = 'mds_' + str(k)
                    print('===> Successfully applied MDS')
                    
                    return pd.DataFrame(mds_features,columns=mds_columns) 
                
                if algo== 'isomap':
                    self.isomap.fit(X_train)
                    isomap_features = self.isomap.transform(data)
                    print('===> Successfully applied ISOMAP')
                    isomap_columns = [None] * isomap_features.shape[1]
                    for k in range(isomap_features.shape[1]):
                        isomap_columns[k] = 'iso_' + str(k)
                    return pd.DataFrame(isomap_features,columns=isomap_columns) 
                if algo=='tsne':
                    
                    tsne_features = self.tsne.fit_transform(data)
                    print('===> Successfully applied t-SNE')
                    tsne_columns = [None] * tsne_features.shape[1]
                    for k in range(tsne_features.shape[1]):
                        tsne_columns[k] = 'tsne_' + str(k)
                    return pd.DataFrame(tsne_features,columns=tsne_columns)  ;

            


    def loadConfiguredModules(self,modules):
        print('-------------------------------')
        print(' STEP : Configured Regression Moduel Loaded')
        print('-------------------------------')
        
        
        self.models = modules
        
        self.configured = True
        
    
    def regressionModelInitialize(self):
        print('-------------------------------')
        print(' STEP : Regression Moduel Initialised')
        print('-------------------------------')
        
        self.models = dict()
        self.params=dict()
        
        
        self.models['knn'] = KNeighborsRegressor()
        self.models['rf'] = RandomForestRegressor(random_state = self.seed)
        self.models['ada'] = AdaBoostRegressor(random_state = self.seed)
        self.models['gb'] = GradientBoostingRegressor(random_state = self.seed)
        self.models['xg'] = XGBRegressor(random_state = self.seed)
        self.models['mlp'] = MLPRegressor()
        self.models['svm'] = SVR()
        self.models['vot'] = VotingRegressor([('knn',self.models['knn']),('ada',self.models['ada']),('rand',self.models['rf']),('svm',self.models['svm'])])
        
        # Preparing parameter for finding optimal parameters
        
        self.params['knn'] ={'n_neighbors':[2,3,4,5],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
        self.params['rf'] = {'max_depth':[2,3,4,5,6],'n_estimators':[50,100,150,200],'min_samples_split':[3,4,5]}
        self.params['ada'] = {'learning_rate':[.01,.001,.0001],'n_estimators':[50,100,150,200],'loss':['linear', 'square', 'exponential']}
        self.params['gb'] = {'learning_rate':[.01,.001,.0001],'n_estimators':[50,100,150,200],'loss':['ls', 'lad', 'huber', 'quantile'],'min_samples_split':[3,4,5]}
        self.params['xg']={'booster':['gbtree', 'gblinear','dart']}
        self.params['mlp']={'solver':['lbfgs','sgd','adam'],'activation':['identity', 'logistic', 'tanh', 'relu'],'hidden_layer_sizes':[(5,5,5),(5,4,3),(10,10,5)]}
        k=['rbf', 'linear','poly','sigmoid']
        c= [1,10,100,.1]
        g=[.0001,.001,.001,.01,.1]
        self.params['svm']=dict(kernel=k, C=c, gamma=g)    
        print('-------------------------------------------')
        print('===> K-Nearest Neighbors initialized')
        print('===> Random Forest initialized')
        print('===> AdaBoost initialized')
        print('===> Gradient Boost initialized')
        print('===> XGBoost initialized')
        print('===> Neural Network initialized')
        print('===> SVM initialized')
        print('===> Voting classifier with KNN, AdaBoost, SVM and Random Forest')
        
        

    
        
    def findParametersAndEvaluate(self,data,strategy,label_name,group=None,dataset=None,cv=5):
        
        
        self.strategy = strategy
        self.results = {}
        
        print('-------------------------------')
        print(' STEP : Finding Parameters & Evaluate Models')
        print('-------------------------------')
        
        self.label_name_check(label_name)
        #print(self.labelset.columns)
        
        # store performance data for each strategy
        
        
        
 
            
        
        
        if (strategy == 'train_test_split' or strategy == 'all'):
            
            self.train_test = dict()
            for model in self.models.keys():
                self.train_test[model] = None
            
            
            
            print('===> Evaluation strategy: Train and Test Split   ')
            X_train, X_test, y_train, y_test = train_test_split(data,self.label_set[label_name],train_size=.7,random_state=self.seed)
            
            print('===> Parameters find-> Start')
            for model in self.models.keys():
                if model == 'vot':
                    continue
                if not self.configured:
                    gd = GridSearchCV(self.models[model],self.params[model],cv=cv,scoring='neg_root_mean_squared_error')
                    gd.fit(X_train,y_train)
                    print('       Parameters for ',model,': ',gd.best_params_)
                    self.models[model] = gd.best_estimator_
                    
                
                
            print('===> Parameters find-> End')   
            
            
            test_performances = dict()
            print('===> Test data performance[RMSE] ')
            for model in self.models.keys():
                self.models[model].fit(X_train,y_train)
                test_performances[model] = mean_squared_error(y_test,self.models[model].predict(X_test),squared=False)
                #print('       Model[',model,']:',test_performances[model])
                self.train_test[model] = test_performances[model]
            print(self.train_test)
        
            self.results['train_test'] = self.train_test
        
        if (strategy == 'cross_val' or strategy == 'all'):
            
            
            self.cross_val = dict()
            cross_val = dict()
            for model in self.models.keys():
                self.cross_val[model] = None
            
            
            print('==============================================')
            print('Evaluation strategy: Cross Validation')
            print('==============================================')
            for model in self.models.keys():

                
                if  model != 'vot' and  not self.configured:
                    print('    ==> Finding params for ',model)
                    gd = GridSearchCV(self.models[model],self.params[model],cv=10,scoring='neg_root_mean_squared_error')
                    gd.fit(data,self.label_set[label_name])
                    print('        Parameters: ',gd.best_params_)
                    self.models[model] = gd.best_estimator_
                
                    
                
                
                cross_val[model] = cross_val_score(self.models[model],data,self.label_set[label_name],scoring='neg_root_mean_squared_error',cv=cv)
                #print('  Score[',model,']:',cross_val_scores[model])
                
                cross_val_mean = -1 * statistics.mean(cross_val[model])
                cross_val_var =  statistics.variance(cross_val[model])
                
                self.cross_val[model] = [cross_val_mean,cross_val_var]
            
            self.results['cross_val'] = self.cross_val
        
        
        if (strategy == 'leave_one_group_out' or strategy == 'all'):
            
            self.leave_group = dict()
            for model in self.models.keys():
                self.leave_group[model] = None
            
            
            
            print('==============================================')
            print('Evaluation strategy: Leave one group out')
            print('==============================================')
            
            logo = LeaveOneGroupOut()
            n_splits = logo.get_n_splits(groups=group)
            
            error= dict()
            
            for model in self.models.keys():
                error[model] = [None]*n_splits
            
            k =0
            for train_index, test_index in logo.split(data,self.label_set[label_name],group):
                #print(test_index)
                
                X_train, y_train = data.iloc[train_index],self.label_set[label_name][train_index]
                X_test, y_test = data.iloc[test_index],self.label_set[label_name][test_index]
                
                for model in self.models.keys():
                   
                    if   model != 'vot' and  not self.configured:
                        
                        print('    ==> Finding params for ',model)
                        gd = GridSearchCV(self.models[model],self.params[model],cv=10,scoring='neg_root_mean_squared_error')
                        gd.fit(X_train,y_train)
                        print('        Parameters: ',gd.best_params_)
                        estimator = gd.best_estimator_
                        
                        
                    
                        self.models[model] = estimator
                        
                        
                    self.models[model].fit(X_train,y_train)
                    error[model][k]  = mean_squared_error(y_test,self.models[model].predict(X_test),squared=False)
                
                    #print('    Model[',model,']:',error[model])
                    
                k = k+1
                
            for model in self.models.keys():
                err_mean = statistics.mean(error[model])
                err_var = statistics.variance(error[model])
                self.leave_group[model] = [err_mean,err_var]
                
                
            self.results['leave_group'] = self.leave_group    
                
                
        if (strategy == 'leave_one_dataset_out' or strategy == 'all'):
            self.leave_dataset = dict()
            for model in self.models.keys():
                self.leave_dataset[model] = None
            
            
            print('==============================================')
            print('Evaluation strategy: Leave one dataset out')
            print('==============================================')
            
            logo = LeaveOneGroupOut()
            n_splits = logo.get_n_splits(groups=dataset)
            
            error= dict()
            
            for model in self.models.keys():
                error[model] = [None]*n_splits
            
            k =0
            
            
            for train_index, test_index in logo.split(data,self.label_set[label_name],dataset):
                
                
                X_train, y_train = data.iloc[train_index],self.label_set[label_name][train_index]
                X_test, y_test = data.iloc[test_index],self.label_set[label_name][test_index]
                
                for model in self.models.keys():
                    
                    
                    if  model != 'vot' and  not self.configured:
                        
                        
                        print('    ==> Finding params for ',model)
                        gd = GridSearchCV(self.models[model],self.params[model],cv=10,scoring='neg_root_mean_squared_error')
                        gd.fit(X_train,y_train)
                        #print('        Parameters: ',gd.best_params_)
                        estimator = gd.best_estimator_
                    
                        self.models[model] = estimator
                        
                        
                    
                    self.models[model].fit(X_train,y_train)
                    
                    error[model][k] = mean_squared_error(y_test,self.models[model].predict(X_test),squared=False)
                
                    #print('    Model[',model,']:',error[model])
                k = k+1
            for model in self.models.keys():
                err_mean = statistics.mean(error[model])
                err_var = statistics.variance(error[model])
                
                self.leave_dataset[model] = [err_mean,err_var]
                
            self.results['leave_dataset'] = self.leave_dataset
                
                
        if (strategy=='sorted_stratified' or strategy == 'all')   :
            
            self.stratified = dict()
            for model in self.models.keys():
                self.stratified[model] = None
            
            # idea from https://scottclowe.com/2016-03-19-stratified-regression-partitions/
            print('==============================================')
            print('Evaluation strategy: Sorted Stratification')
            print('==============================================')
            
            label_df = pd.DataFrame(self.label_set)
            
            indices = label_df.sort_values(by=[label_name]).index.tolist()
            splits = dict()
            
            error = dict()
            for model in self.models.keys():
                error[model] = [None]*cv
            
            for i in range(cv):
                splits[i] = list()
                
            for i in range(len(indices)):
                if i%cv == 0:
                    pick = random.sample(range(cv),cv)
                cur_pick = pick.pop()
                splits[cur_pick].append(indices[i])
                
            for i in range(cv):
                test_index = splits[i]
                train_index = []
                for j in range(cv):
                    if j != i:
                        train_index = train_index + splits[j]
                
                ##########################################
                
                # Code to training model on sorted stratified set
                X_train, y_train = data.iloc[train_index],self.label_set[label_name][train_index]
                X_test, y_test = data.iloc[test_index],self.label_set[label_name][test_index]
                
                
                for model in self.models.keys():
                    if  model != 'vot' and  not self.configured:
                        
                        print('    ==> Finding params for ',model)
                        gd = GridSearchCV(self.models[model],self.params[model],cv=10,scoring='neg_root_mean_squared_error')
                        gd.fit(X_train,y_train)
                        print('        Parameters: ',gd.best_params_)
                        estimator = gd.best_estimator_
                        self.models[model] = estimator
                        
                        
                        
                    
                    self.models[model].fit(X_train,y_train)
                    
                    error[model][i] = mean_squared_error(y_test,self.models[model].predict(X_test),squared=False)
                
                    #print('    Model[',model,']:',error[model])
                
            
            for model in self.models.keys():
                err_mean = statistics.mean(error[model])
                err_var = statistics.variance(error[model])
                self.stratified[model] = [err_mean,err_var]
                ##########################################
            
            self.results['stratified'] = self.stratified
             
        else:
            print('Unsupported evaluation strategy')
            return None
    
        return self.results
    
        # Preparing dataframe with results for report generation
        
        """        
        if strategy == 'train_test_split':
            df = pd.DataFrame(columns = ['model','train_test])
            for model in self.models.keys():
                df = df.append({'model':model,'train_test':self.train_test[model]},ignore_index=True)

        if strategy == 'cross_val':
            df = pd.DataFrame(columns = ['model','train_test_mean','train_test_var'])
            for model in self.models.keys():
                df = df.append({'model':model,'train_test_mean':self.train_test[model][0],'train_test_var':self.train_test[model][1]},ignore_index=True)

        if strategy == 'leave_one_group_out':
            df = pd.DataFrame(columns = ['model','train_test_mean','train_test_var'])
            for model in self.models.keys():
                df = df.append({'model':model,'train_test_mean':self.train_test[model][0],'train_test_var':self.train_test[model][1]},ignore_index=True)
        
        if strategy == 'leave_one_dataset_out':
            df = pd.DataFrame(columns = ['model','train_test_mean','train_test_var'])
            for model in self.models.keys():
                df = df.append({'model':model,'train_test_mean':self.train_test[model][0],'train_test_var':self.train_test[model][1]},ignore_index=True)
        
        if strategy == 'sorted_stratified':
            df = pd.DataFrame(columns = ['model','train_test_mean','train_test_var'])
            for model in self.models.keys():
                df = df.append({'model':model,'train_test_mean':self.train_test[model][0],'train_test_var':self.train_test[model][1]},ignore_index=True)



    
        if strategy == 'all':
            df = pd.DataFrame(columns = ['model','train_test','cross_val','leave_group','leave_dataset','stratified'])
            for model in self.models.keys():
                df = df.append({'model':model,'train_test':self.train_test[model],'cross_val':self.cross_val[model],'leave_group':self.leave_group[model],'leave_dataset':self.leave_dataset[model],'stratified':self.stratified[model]},ignore_index=True)

        return df            
    
        """    
    
    def report(self,currentOutput,report_name=''):
        
        
        df = pd.DataFrame(columns = ['model','train_test','cross_val_mean','cross_val_var','leave_group_mean','leave_group_var','leave_dataset_mean','leave_dataset_var','stratified_mean','stratified_var'])
        for model in self.models.keys():
            
            df = df.append({'model':model,'train_test':self.train_test[model],'cross_val_mean':self.cross_val[model][0],'cross_val_var':self.cross_val[model][1],'leave_group_mean':self.leave_group[model][0],'leave_group_var':self.leave_group[model][1],'leave_dataset_mean':self.leave_dataset[model][0],'leave_dataset_var':self.leave_dataset[model][1],'stratified_mean':self.stratified[model][0],'stratified_var':self.stratified[model][1]},ignore_index=True)
            
        filename = report_name 
            
        df.to_csv(filename,index=False)
        print('==============================================')
        print(' Report Generation')
        print('==============================================')
        print(' ===> Successfully generated ')
        print(' ===> Results saved in easyRegress_report.csv file')
        

    
    def activateGroups(self,groups):
        self.groups = groups
        self.flagGroup = True
        
    def activateDatasets(self,datasets):
        self.datasets = datasets
        self.flagDataset = True
        
    def activateLabel(self,label):
        self.label = label
        
     
    def buildPipeline(self,sequence,report_name=''):
        """
        <feature_name> : Name of feature
        feature_extraction: Apply feature extraction based on correlation
        feature_scaling: Apply feature scaling. Options: Standard, MinMax
        feature_fusion:  Apply feature fusion. Options: gini, entropy, pca, isomap, mds, tsne
        load_models: Load regression models.
        find_evaluate: Model evaluation. Options: train_test_split, cross_validation, leave_one_group_out, leave_one_dataset_out, sorted_stratified
        report_results: Report results. Options: table, chart
      

        """
        currentOutput = None
        
        for index, step in enumerate(sequence):
            
            
            label = self.label
            groups = self.groups
            datasets = self.datasets
            
            if index == 0:
                self.feature_name_check(step)
                currentOutput = self.feature_set[step][2]
                
            
            elif step == 'feature_extraction':
                
                results = self.extractFeatures(currentOutput)
                currentOutput = results
            elif step == 'feature_scaling_std':
                print(currentOutput.shape)
                results = self.Scaling(currentOutput,'std')
                currentOutput = results
            elif step == 'feature_scaling_mmax':
                
                results = self.Scaling(currentOutput,'mmax')
                currentOutput = results
            elif step == 'feature_fusion_pca':
                
                results = self.DimRed('pca',currentOutput,{'n_components':2,'n_neighbors':3})
                currentOutput = results
            elif step == 'feature_fusion_mds':
                
                results = self.DimRed('mds',currentOutput,{'n_components':2,'n_neighbors':3})
                currentOutput = results
            elif step == 'feature_fusion_isomap':
                
                results = self.DimRed('isomap',currentOutput,{'n_components':2,'n_neighbors':3})
                currentOutput = results
            elif step == 'feature_fusion_tsne':
                
                results = self.DimRed('tsne',currentOutput,{'n_components':2,'n_neighbors':3})
                currentOutput = results
            elif step == 'feature_fusion_entropy':
                results = self.getEntropy(currentOutput)
                currentOutput = results
                print(results)               
            elif step == 'feature_fusion_gini':
                results = self.getGINI(currentOutput)
                currentOutput = results
                print(results)
                
            elif step == 'load_modules':
                self.regressionModelInitialize()
            
            elif step == 'evaluate_train_test':
                if label == None:
                    print(' ====> Error: labels are not loaded')
                  
                results =self.findParametersAndEvaluate(currentOutput,'train_test_split',label)
                currentOutput = results
            elif step == 'evaluate_cross_val':
                if label == None:
                    print(' ====> Error: labels are not loaded')
                
                results =self.findParametersAndEvaluate(currentOutput,'cross_val',label)
                currentOutput = results
            elif step == 'evaluate_leave_group_out':
                if label == None:
                    print(' ====> Error: labels are not loaded')
                
                if self.flagDataset == False:
                    print(' ====> Error: groups ids are not loaded')
                results =self.findParametersAndEvaluate(currentOutput,'leave_one_group_out',label,group=groups)
                currentOutput = results
            elif step == 'evaluate_leave_dataset_out':
                if label == None:
                    print(' ====> Error: labels are not loaded')
                
                if self.flagDataset == False:
                    print(' ====> Error: datasets ids are not loaded')
                results =self.findParametersAndEvaluate(currentOutput,'leave_one_dataset_out',label,dataset = datasets)
                currentOutput = results
            elif step == 'evaluate_stratified':
                if label == None:
                    print(' ====> Error: labels are not loaded')
                results =self.findParametersAndEvaluate(currentOutput,'sorted_stratified',label)
                currentOutput = results
            elif step == 'all':
                if label == None:
                    print(' ====> Error: labels are not loaded')
                results =self.findParametersAndEvaluate(currentOutput,'all',label,group = groups, dataset = datasets)
                currentOutput = results
                
            elif step == 'report_csv':
                self.report(currentOutput,report_name)
            else:
                print(' Unsupported module ',step,' is specified')
                