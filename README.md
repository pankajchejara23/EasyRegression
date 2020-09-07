# Easy Regression


Easy Regression is a python package which simplify the multimodal data analysis pipeline for regression models. It supports following features 

  - Feature extraction on basis of correlation
  - Feature fusion (Entropy, Gini, PCA, Isomap, MDS, t-SNE)
  - Feature standardisation (Standard and Min-Max scaling)
  - Regression models (K-Nearest Neighbors, Random Forest, AdaBoost, XGBoost, Gradient Boost, SVM, Voting)
  - Parameters searching using GridSearchCV form Sci-kit learn
  - Model evaluation (train test split, cross validation, leave one group out, leave one dataset out, sorted stratification)
  - Report module (under development)
  
## Usage
EasyRegression allows loading of multiple data files. It also supports the type of features e.g., individual subject's feature or group's feature. For instance, in the setting of collaborative learning, data are collected at individual level and group level. EasyRegression supports loading of both features.

``` python
import EasyRegression as es

obj = es.EasyRegression()

# Load feature file 
obj.loadFeatures('open_smile_num.csv','grp','Open_smile')

# Load label file
obj.loadLabels('JLA_dataset_codes.csv')
```
Output:
```
 _____                    
| ____|  __ _  ___  _   _ 
|  _|   / _` |/ __|| | | |
| |___ | (_| |\__ \| |_| |
|_____| \__,_||___/ \__, |
                    |___/ 

 ____                                      _               
|  _ \   ___   __ _  _ __   ___  ___  ___ (_)  ___   _ __  
| |_) | / _ \ / _` || '__| / _ \/ __|/ __|| | / _ \ | '_ \ 
|  _ < |  __/| (_| || |   |  __/\__ \\__ \| || (_) || | | |
|_| \_\ \___| \__, ||_|    \___||___/|___/|_| \___/ |_| |_|
              |___/                                        

-------------------------------
 STEP : Features loading
-------------------------------
NoneType: None
===> Feature file: open_smile_num.csv  is loaded successfully !
===> Summary:
     #instances: 306
     #attributes: 1582
     #numeric-attributes: 1582

===> Label file: JLA_dataset_codes.csv  is loaded successfully !
===> Summary:
     #labels: 8
     labels: ['SMU', 'CF', 'KE', 'ARG', 'STR', 'CO', 'ITO', 'final']
```
Once the data and labels are loaded, you need to activate the label which you want to use in the regression analysis as target variable. 
As it can be seen that in our label file there are multiple labels therefore we will specify the target variable now.
```python
# Activate `final` variable as target variable
obj.activateLabel('final')
```
Now, we will build our pipeline for regression analysis. Here, we need to specify operations that we want to perform on the data.
```python
# Build pipeline
obj.buildPipeline(['Open_smile','feature_scaling_std','feature_fusion_pca','load_modules','evaluate_train_test'])

```
 ## Complete list of modules
 
- `feature_extraction` : this module will extract and return uncorrelated features.
- `feature_scaling_std`: this module will standardise the data.
- `feature_scaling_mmax`: this module will perform min-max scaling
- `feature_fusion_pca`: this module apply PCA and return PCA-components that explains 90% or more variance in the data.
- `feature_fusion_mds`: this module applies MDS dimensionality reduction.
- `feature_fusion_isomap`: this module applies Isomap dimensionality reduction.
- `feature_fusion_tsne`: this module applies t-SNE dimensionality reduction.
- `feature_entropy`: Under development
- `feature_gini`: Under development
- `load_modules`: Load pre-defined regression models.
- `evaluate_train_test`: Find parameters for regression models and perform model validation using train and test split.
- `evaluate_cross_val`: Find parameters for regression models and perform model validation using cross validation (default 10-fold).
- `evaluate_leave_one_group_out`: Find parameters for regression models and perform model validation using leave one group out.
- `evaluate_leave_one_dataset_out`: Find parameters for regression models and perform model validation using leave one dataset out.
- `evaluate_stratified`: Find parameters for regression models and perform model validation using sorted stratification.

## Future development
At the moment, the configuration options are not supported for above mentioned modules. The support will be added later.

 

- 
