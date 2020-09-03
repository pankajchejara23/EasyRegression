
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import seaborn as sns
from sklearn.model_selection import train_test_split
import math

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars, BayesianRidge, SGDRegressor, RANSACRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
#from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
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




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
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
                return self.std.fit_transform(data)
            elif algo == 'mmax':
                return self.mmax.fit_transform(data)
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

                print(self.pca_features)
        
        
            
            
        
        
        

