
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import seaborn as sns
from sklearn.model_selection import train_test_split
import math
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import RANSACRegressor

from sklearn.model_selection import cross_val_score

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


class EasyRegression:
    def __init__(self):
        print('EasyRegression is initialized.')
        self.dataReady = False
        
        # Scalers
        self.std = StandardScaler()
        self.mmax = MinMaxScaler()
        
        self.random_state = 42
        

        

    """
    loaddata: function to load input data files
    """
    def loadData(self,features_file,labels_file):
        try:
            self.features = pd.read_csv(features_file)
            self.labels = pd.read_csv(labels_file)
            print('Input files are loaded successfully.')
            # Check for data consistency
            features_shape = self.features.shape[0]
            labels_shape = self.labels.shape[0]

            if features_shape != labels_shape:
                print('Features and labels are not same. Check your files.')
            else:
                print('Cosntraint-DataSize Check: Passed')
                self.dataReady = True
                print('  # instances :', self.features.shape[0])
                print('  # attributes:',self.features.shape[1])


        except FileNotFoundError:
            print('Specified file do not exists')
    def getBasicFeatures(self):
        return self.features
    
    def getLabels(self):
        return self.labels

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
    
    """
        This function performs group-level feature computation
        supported fusions: Dimensionality reduction, Entropy, Gini, Average
    """
    def getGroupFeatures(self):
        group_feature_labels = ['add','del','speak','turns']
        self.features_group = dict()
        # iterate for each group-level feature 
        for grp_feature in group_feature_labels:
            
            tmp = list()
            
            # get all column names similar to grp_feature
            for indiv_feature in self.features.columns:
                if grp_feature in indiv_feature:
                    tmp.append(indiv_feature)
            
            self.features_group[grp_feature] = tmp.copy()
        
        print(self.features_group)
        
    
    # preparing gini coefficient
    def getGINI(self):
        """Calculate the Gini coefficient of a numpy array."""
        
        self.getGroupFeatures()
        gini = dict()
        for key in self.features_group.keys():
 
            
            tmp = self.features[self.features_group[key]].values
            tmp = tmp + 0.0000001 
            tmp = np.sort(tmp)
            index =  np.arange(1,tmp.shape[1]+1) 
            n = tmp.shape[1]
            

            
            key = 'grp_gini_'+key
            gini[key] = ((np.sum((2 * index - n  - 1) * tmp,axis=1)) / (n * np.sum(tmp,axis=1))) #Gini coefficient
        
        self.gini_features = pd.DataFrame(gini)
        return self.gini_features
    


    # Compute entropy features for individual features
    def getEntropy(self):
        self.getGroupFeatures()
        entropy = dict()
        for key in self.features_group.keys():      
            tmp = self.features[self.features_group[key]].values
            tmp = tmp
            tmp_sum = tmp.sum(axis=1,keepdims=True) + .0000000000001
            p = tmp/tmp_sum
            
            key = 'grp_entropy_'+key
            entropy[key] = entr(p).sum(axis=1)/np.log(2)
  
        self.entropy_features = pd.DataFrame(entropy)
        return self.entropy_features
    
    
    """
    Apply dimentionality reduction on features
    PCA
    
    """
    def Scaling(self,data,algo):
        if algo in ['std','mmax']:
            if algo == 'std':
                return pd.DataFrame(self.std.fit_transform(data), columns=data.columns)
            elif algo == 'mmax':
                return pd.DataFrame(self.mmax.fit_transform(data), columns=data.columns)
            else:
                print('Unsupported scaler')
                return None
            
    def DimRed(self,algo,data,params):
        if algo not in ['pca','mds','isomap','t-sne']:
            print('Unsupported dimension reduction algorithm specified')
            return None
        else:
            if algo!='pca' and len(params) ==0:
                print('Specify n_components/n_neighbors parameters')
                return None
            else:    
                # Dimensionality reduction
    
                self.pca = PCA()
                self.mds = manifold.MDS(n_components=params['n_components'],max_iter=100,n_init=1)
                self.isomap = manifold.Isomap(n_neighbors=params['n_neighbors'],n_components=params['n_components'])
                self.tsne = manifold.TSNE(n_components=params['n_components'],init='pca',random_state=self.random_state)
            
                self.pca_features = self.pca.fit_transform(data)
                self.mds_features = self.mds.fit_transform(data)
                self.isomap_features = self.isomap.fit_transform(data)
                self.tsne_features = self.tsne.fit_transform(data)

    
    def regressionModelInitialize(self):
        print('    Preparating Regression Models')
        self.models = dict()
        self.params=dict()
        
        
        self.models['knn'] = KNeighborsRegressor()
        self.models['rf'] = RandomForestRegressor()
        self.models['ada'] = AdaBoostRegressor()
        self.models['gb'] = GradientBoostingRegressor()
        self.models['xg'] = XGBRegressor()
        self.models['mlp'] = MLPRegressor()
        self.models['svm'] = SVR()
        self.models['vot'] = VotingRegressor([('knn',self.models['knn']),('ada',self.models['ada']),('rand',self.models['rf']),('svm',self.models['svm'])])
        
        # Preparing parameter for finding optimal parameters
        
        self.params['knn'] ={'n_neighbors':[2,3,4,5],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
        self.params['rf'] = {'max_depth':[2,3,4,5,6],'n_estimators':[50,100,150,200],'min_samples_split':[3,4,5]}
        self.params['ada'] = {'learning_rate':[.01,.001,.0001],'n_estimators':[50,100,150,200],'loss':['linear', 'square', 'exponential']}
        self.params['gb'] = {'learning_rate':[.01,.001,.0001],'n_estimators':[50,100,150,200],'loss':['ls', 'lad', 'huber', 'quantile'],'min_samples_split':[3,4,5]}
        self.params['xg']={'booster':['gbtree', 'gblinear','dart'],'max_depth':[2,3,4,5,6]}
        self.params['mlp']={'solver':['lbfgs','sgd','adam'],'activation':['identity', 'logistic', 'tanh', 'relu'],'hidden_layer_sizes':[(5,5),(5,5,5),(5,4,3),(10,10,5)],'max_iter':[1000]}
        k=['rbf', 'linear','poly','sigmoid']
        c= [1,10,100,.1]
        g=[.0001,.001,.001,.01,.1]
        self.params['svm']=dict(kernel=k, C=c, gamma=g)    
        print('-------------------------------------------')
        print('  K-Nearest Neighbors initialized')
        print('  Random Forest initialized')
        print('  AdaBoost initialized')
        print('  Gradient Boost initialized')
        print('  XGBoost initialized')
        print('  Neural Network initialized')
        print('  SVM initialized')
        print('  Voting classifier with KNN, AdaBoost, and Random Forest')
        print('-------------------------------------------')
        
        
    def findParameters(self,strategy,label_name,group,cv=10):
        if label_name not in self.labels.columns:
            print('Label does not exists')
            return None
        self.scaled_features = self.Scaling(self.features,'std')
        
        if (strategy == 'train_test_split'):
            print('==============================================')
            print('  Evaluation strategy: Train and Test Split   ')
            print('==============================================')
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.scaled_features,self.labels[label_name])
            
        
            for model in self.models.keys():
                if model == 'vot':
                    continue
                print('    ==> Finding params for ',model)
                gd = GridSearchCV(self.models[model],self.params[model],cv=10,scoring='neg_root_mean_squared_error')
                gd.fit(self.X_train,self.y_train)
                print('        Parameters: ',gd.best_params_)
                self.models[model] = gd.best_estimator_
        elif (strategy == 'cross_val'):
            self.cross_val_scores = dict()
            print('==============================================')
            print('Evaluation strategy: Cross Validation')
            print('==============================================')
            for model in self.models.keys():
                if model == 'vot':
                    continue
                print('    ==> Finding params for ',model)
                gd = GridSearchCV(self.models[model],self.params[model],cv=10,scoring='neg_root_mean_squared_error')
                gd.fit(self.scaled_features,self.labels[label_name])
                print('        Parameters: ',gd.best_params_)
                self.models[model] = gd.best_estimator_
                
                self.cross_val_scores[model] = cross_val_score(self.models[model],self.features,self.labels[label_name],scoring='neg_root_mean_squared_error')
                print('  Score[',model,']:',self.cross_val_scores[model])
        elif (strategy == 'leave_one_group_out'):
            print('==============================================')
            print('Evaluation strategy: Leave one group out')
            print('==============================================')
            
            logo = LeaveOneGroupOut()
            logo.get_n_splits(groups=group)
            
            
            for train_index, test_index in logo.split(self.scaled_features,self.labels[label_name],group):
                #print(test_index)
                
                X_train, y_train = self.scaled_features.iloc[train_index],self.labels[label_name][train_index]
                X_test, y_test = self.scaled_features.iloc[test_index],self.labels[label_name][test_index]
                
                for model in self.models.keys():
                    if model == 'vot':
                        continue
                    print('    ==> Finding params for ',model)
                    gd = GridSearchCV(self.models[model],self.params[model],cv=10,scoring='neg_root_mean_squared_error')
                    gd.fit(X_train,y_train)
                    print('        Parameters: ',gd.best_params_)
                    estimator = gd.best_estimator_
                    
                    estimator.fit(X_train,y_train)
                    
                    error = mean_squared_error(y_test,estimator.predict(X_test),squared=False)
                
                    print('    Model[',model,']:',error)
                
                
                
        elif (strategy == 'leave_one_dataset_out'):
            print('==============================================')
            print('Evaluation strategy: Leave one dataset out')
            print('==============================================')
            
            logo = LeaveOneGroupOut()
            logo.get_n_splits(groups=group)
            
            
            for train_index, test_index in logo.split(self.scaled_features,self.labels[label_name],group):
                #print(test_index)
                
                X_train, y_train = self.scaled_features.iloc[train_index],self.labels[label_name][train_index]
                X_test, y_test = self.scaled_features.iloc[test_index],self.labels[label_name][test_index]
                
                for model in self.models.keys():
                    if model == 'vot':
                        continue
                    print('    ==> Finding params for ',model)
                    gd = GridSearchCV(self.models[model],self.params[model],cv=10,scoring='neg_root_mean_squared_error')
                    gd.fit(X_train,y_train)
                    print('        Parameters: ',gd.best_params_)
                    estimator = gd.best_estimator_
                    
                    estimator.fit(X_train,y_train)
                    
                    error = mean_squared_error(y_test,estimator.predict(X_test),squared=False)
                
                    print('    Model[',model,']:',error)
                
        elif (strategy=='sorted_stratified')   :
            # idea from https://scottclowe.com/2016-03-19-stratified-regression-partitions/
            indices = self.labels.sort_values(by=self.labels[label_name]).index.tolist()
            splits = dict()
            
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
                
                ##########################################
                
            
                
                
            
        else:
            print('Unsupported evaluation strategy')
            return None
        
    def testPerformance(self):
        self.test_performances = dict()
        print('    ==> Performance on test data ')
        for model in self.models.keys():
            self.models[model].fit(self.X_train,self.y_train)
            self.test_performances[model] = mean_squared_error(self.y_test,self.models[model].predict(self.X_test),squared=False)
            print('   Model[',model,']:',self.test_performances[model])

