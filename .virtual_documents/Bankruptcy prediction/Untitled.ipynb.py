# Supressing warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Libraries for data manipulation, statistical operations and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")

from scipy.io import arff                                                                  # to load .arff files
import missingno as msno                                                                   # to analyze type of missing data
import fancyimpute                                                                         # For performing k-NN and MICE imputations
import impyute as impy                                                                     # For performing Expectation Maximization (EM) imputation
from sklearn.impute import SimpleImputer                                                   # For performing Mean imputation
from sklearn.impute import IterativeImputer                                                # Using this in place of MICE (Multiple Imputation by Chained Equations)

from sklearn.model_selection import KFold                                                  # For performing kFold cross-validation
from collections import Counter                                                            # For counting class objects ( elements as keys, their count as values)
from collections import OrderedDict                                                        # For dictionary operations
from imblearn.over_sampling import SMOTE                                                   # To perform SMOTE oversampling while dealing with imbalanced data

import sklearn
from xgboost import XGBClassifier                                                          # Importing classification models for modeling
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import random

# All metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve


# Loads the 5 raw .arff files into a list
def load_arff():
    n = 5
    return [arff.loadarff('data/' + str(i+1) + 'year.arff') for i in range(n)]

# Loads the 5 raw .arff files into pandas dataframes
def init_dataframes():
    return [pd.DataFrame(i_year[0]) for i_year in load_arff()]

# Sets column headers as X1,...X64 and label as Y for all 5 dataframes
def set_new_headers(dataframes):
    cols = ['X' + str(i+1) for i in range(len(dataframes[0].columns)-1)]
    cols.append('Y')
    for df in dataframes:
        df.columns = cols

# Objects are a list of pandas dataframes for the 5 year datafiles
dataframes = init_dataframes()

# Sets new headers for the dataframes with renamed set of features (X1 to X64) and a label (Y)
set_new_headers(dataframes)

# print the first 5 rows of a dataset 'year1'
dataframes[0].head()


column_list = [dataframes[0].columns]
column_list


def convert_to_float(df):
    for i in range(5):
        index = 1
        while index <= 63:
            colname = df[i].columns[index]
            col = getattr(df[i], colname)
            df[i][colname] = col.astype(float)
            index += 1
convert_to_float(dataframes)


def convert_classlabel(df):
    for i in range(len(df)):
        col = getattr(df[i], 'Y')
        df[i]['Y'] = col.astype(int)
        
convert_classlabel(dataframes)


""" Note that our 'dataframes' is actually a list of dataframes for all five years of data"""

class Data:
    
    """ All basic operations we perform on the dataset are created under this class"""
    
    def __init__(self, dataframes):
        """ Initializes the class with dataframes as argument"""
        self.dataframes = dataframes
        
    def drop_missing(self, verbose=False):
        """ Drops the missing values and shows the effects of doing so in numbers"""
        cleaned = [df.dropna(axis=0, how='any') for df in self.dataframes]
        if verbose:
            for i in range(len(self.dataframes)):
                print(str(i+1) + 'year:', 'Original length=', len(self.dataframes[i]), '\tCleaned length = ', len(cleaned[i]), '\tMissing data = ', len(self.dataframes[i] - cleaned[i]))
        else:
            return cleaned
    
    def generate_sparsity_matrix(self):
        """ Generates Sparsity Matrix for the missing data in all dataframes"""
        for i in range(5):
            missing_values = self.dataframes[i].columns[self.dataframes[i].isnull().any()].tolist() # Making a list of all missing values in all 5 dataframes
            msno.matrix(self.dataframes[i][missing_values], figsize=(20,5))
    
    def generate_heatmaps_missing_values(self):
        """ Generates heatmaps for missing values in all dataframes and displays respective correlations among features in a ascending tabular form"""
        for i in range(5):
            missing_values = self.dataframes[i].columns[self.dataframes[i].isnull().any()].tolist() # Making a list of all missing values in all 5 dataframes
            msno.heatmap(self.dataframes[i][missing_values], figsize=(20,20))
    
    def do_mean_imputation(self):
        """ Performs mean imputation method on missing values on all dataframes passed as argument and returns a list of dataframes of same length as of argument"""
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputed_dataframes = [pd.DataFrame(imputer.fit_transform(dfs)) for dfs in self.dataframes]
        for i in range(len(self.dataframes)):
            imputed_dataframes[i].columns = self.dataframes[i].columns
        return imputed_dataframes
    
    def do_knn_imputation(self):
        """ Performs KNN imputation on missing values in all dataframes passed as argument and returns a list of dataframes of same length as of argument"""
        imputed_dataframes = [fancyimpute.KNN(k=100, verbose=True).fit_transform(self.dataframes[i]) for i in range(len(self.dataframes))]
        return [pd.DataFrame(data = imputed_dataframes[i]) for i in range(len(self.dataframes))]
    
    def do_EM_imputation(self):
        """ Performs Expectation Maximization imputation on all dataframes passed as argument and returns a list of dataframes of same length as of argument"""
        imputed_dataframes = [impy.imputation.cs.em(self.dataframes[i].values, loops = 50, dtype = 'cont') for i in range(len(self.dataframes))]
        return [pd.DataFrame(data = imputed_dataframes[i]) for i in range(len(self.dataframes))]
    
    def do_MICE_imputation(self):
        """ Using Sklearn's Iterative Imputer which is inspired by MICE imputer as unable to import MICE imputer module"""
        """ Performs MICE (Multiple Imputation from Chained Equations) on all dataframes passed as argument and returns a list of dataframes of same length as of argument"""
        imputed_dataframes = [IterativeImputer(max_iter=10, random_state=10, verbose=False).fit_transform(self.dataframes[i]) for i in range(len(self.dataframes))]
        return [pd.DataFrame(data = imputed_dataframes[i]) for i in range(len(self.dataframes))]
           
    def set_header(self):
        """ Sets column names for all columns as X1, X2...X63 for all features and Y for label"""
        columns = ['X' + str(i+1) for i in range(len(self.dataframes[0].columns)-1)]
        columns.append('Y')
        for df in self.dataframes:
            df.columns = columns



# Doing a quick analysis of how many missing values are there in each of the 5 dataframes
Data = Data(dataframes)


Data.drop_missing(verbose=True)


Data.generate_sparsity_matrix()


Data.generate_heatmaps_missing_values()


m = Data.do_mean_imputation()


knn = Data.do_knn_imputation()
set_new_headers(knn)


em = Data.do_EM_imputation()
set_new_headers(em)


mice = Data.do_MICE_imputation()
set_new_headers(em)


imputed_dataframes_dictionary = OrderedDict()
imputed_dataframes_dictionary['Mean'] = m
imputed_dataframes_dictionary['k-NN'] = knn
imputed_dataframes_dictionary['EM'] = em
imputed_dataframes_dictionary['MICE'] = mice


def data_imbalance(dataframes):
    for i in range(len(dataframes)):
        print('Dataset: '+str(i+1)+' ndyear')
        print(dataframes[i].groupby('Y').size())
        minority_percent = (dataframes[i]['Y'].tolist().count(1) / len(dataframes[i]['Y'].tolist()))*100
        print('Minority (label 1) percentage: '+  str(minority_percent) + 'get_ipython().run_line_magic("')", "")
        print('-'*64)
        
data_imbalance(dataframes)


class OVERSAMPLING:
    
    """ Contains all methods required for performing SMOTE oversampling.........__init__(self) not required"""

    def split_dataframes_features_labels(self,dfs):
        """ Split the features and labels into separate dataframes for all the original dataframes """ 
        feature_dataframes = [dfs[i].iloc[:,0:64] for i in range(len(dfs))]
        label_dataframes = [dfs[i].iloc[:,64] for i in range(len(dfs))]
        return feature_dataframes, label_dataframes    
    
    
    def oversample_data_SMOTE(self,dfs, verbose=False):
        """ Performs oversampling for dataframes passed in as an argument"""
        smote = SMOTE('auto' , random_state=42, k_neighbors=10)
        #Split the features and labels for each dataframe
        feature_dataframes, label_dataframes = self.split_dataframes_features_labels(dfs)
        resampled_feature_arrays = []
        resampled_label_arrays = []
        for i in range(len(dfs)):
            if verbose: print('Dataset: ' + str(i+1) + 'year:')
            if verbose: print('Original dataset shape {}'.format(Counter(label_dataframes[i])))
            dfi_features_res, dfi_label_res = smote.fit_sample(feature_dataframes[i], label_dataframes[i])
            if verbose: print('Resampled dataset shape {}\n'.format(Counter(dfi_label_res)))
            # Append the resampled feature and label arrays of ith dataframe to their respective list of arrays    
            resampled_feature_arrays.append(dfi_features_res)
            resampled_label_arrays.append(dfi_label_res)        
        return resampled_feature_arrays, resampled_label_arrays

    
    def restructure_arrays_to_dataframes(self, feature_arrays, label_arrays):  
        """ Utility Function to convert the arrays of features and labels to pandas dataframes, and then join them. Also re-assign the columns headers """
        resampled_dfs = []
        for i in range(len(feature_arrays)):
            feature_df = pd.DataFrame(data=feature_arrays[i])
            label_df = pd.DataFrame(data=label_arrays[i])
            # Must set the column header for label_df, otherwise it wont join with feature_df, as columns overlap (with col names '0')
            label_df.columns=['Y'] 
            resampled_dfs.append(feature_df.join(label_df))
        # re-assign the column headers for features and labels    
        self.set_headers(resampled_dfs)
        return resampled_dfs

    def set_headers(self,dataframes):
        cols = ['X' + str(i+1) for i in range(len(dataframes[0].columns)-1)]
        cols.append('Y')
        for df in dataframes:
            df.columns = cols


    def perform_oversampling_on_imputed_dataframes(self, df_dict):
        """ Performs SMOTE on all imputed dataframes and stores them in a dictionary"""
        imputed_oversampled_dataframes_dictionary = OrderedDict()
        for key,dfs in df_dict.items():
            print('SMOTE Oversampling for ' + key + ' imputed dataframes\n')
            smote_feature_arrays, smote_label_arrays = self.oversample_data_SMOTE(dfs, verbose=True)
            oversampled_dataframes = self.restructure_arrays_to_dataframes(smote_feature_arrays, smote_label_arrays)
            imputed_oversampled_dataframes_dictionary[key] = oversampled_dataframes
            print('-'*100)
        return imputed_oversampled_dataframes_dictionary



imputed_oversampled_dataframes_dictionary = OVERSAMPLING().perform_oversampling_on_imputed_dataframes(imputed_dataframes_dictionary)


def prepare_kfold_cv_data(k, X, y, verbose=False):
    X = X.values
    y = y.values
    kf = KFold(n_splits=k, shuffle=False, random_state=42)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for train_index, test_index in kf.split(X):
        X_train.append(X[train_index])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])
    return X_train, y_train, X_test, y_test


# Gaussian Naive Bayes classifier
gnb_classifier = GaussianNB()


# Logistic Regression classifier
lr_classifier = LogisticRegression(penalty = 'l1', random_state = 0, solver='liblinear')


# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)


# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators = 5, criterion = 'entropy')


# eXtreme Gradient Boosting Classifier (XGBClassifier)
xgb_classifier = XGBClassifier()


# Balanced Bagging Classifier
bb_classifier = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(criterion='entropy'), n_estimators = 5, bootstrap = True)


# creating a dictionary of models
models_dictionary = OrderedDict()

models_dictionary['Gaussian Naive Bayes'] = gnb_classifier
models_dictionary['Logistic Regression'] = lr_classifier
models_dictionary['Decision Tree'] = dt_classifier
models_dictionary['Extreme Gradient Boosting'] = xgb_classifier
models_dictionary['Random Forest'] = rf_classifier
models_dictionary['Balanced Bagging'] = bb_classifier


# perform data modeling
def perform_data_modeling(_models_, _imputers_, verbose=False, k_folds=5):
    
    # 7 Models
    # 4 Imputers
    # 5 datasets (for 5 years)
    # 7 metrics, averaged over all the K-Folds
    model_results = OrderedDict()
    
    # Iterate over the models
    for model_name, clf in _models_.items():
        if verbose: print("-"*120, "\n", "Model: " + '\033[1m' + model_name + '\033[0m' + " Classifier")
        imputer_results = OrderedDict()
        
        # Iterate over the different imputed_data mechanisms (Mean, k-NN, EM, MICE)
        for imputer_name, dataframes_list in _imputers_.items():
            if verbose: print('\tImputer Technique: ' + '\033[1m' + imputer_name + '\033[0m')
            
            # call the split_dataframes_features_labels function to get a list of features and labels for all the dataframes
            feature_dfs, label_dfs = split_dataframes_features_labels(dataframes_list)            
            
            year_results = OrderedDict()
            
            # Iterate over dataframe_list individually
            for df_index in range(len(dataframes_list)):
                if verbose: print('\t\tDataset: ' + '\033[1m' + str(df_index+1) + 'year' + '\033[0m')
                
                # Calling the 'prepare_kfold_cv_data' returns lists of features and labels 
                # for train and test sets respectively.
                # The number of items in the list is equal to k_folds
                X_train_list, y_train_list, X_test_list, y_test_list = prepare_kfold_cv_data(k_folds, feature_dfs[df_index], label_dfs[df_index], verbose)
                
                metrics_results = OrderedDict()
                accuracy_list = np.zeros([k_folds])
                precision_list = np.zeros([k_folds,2])
                recall_list = np.zeros([k_folds,2])
                TN_list = np.zeros([k_folds])
                FP_list = np.zeros([k_folds])
                FN_list = np.zeros([k_folds])
                TP_list = np.zeros([k_folds])                
                
                # Iterate over all the k-folds
                for k_index in range(k_folds):
                    X_train = X_train_list[k_index]
                    y_train = y_train_list[k_index]
                    X_test = X_test_list[k_index]
                    y_test = y_test_list[k_index]
                    
                    # Fit the model and 
                    clf = clf.fit(X_train, y_train)
                    y_test_predicted = clf.predict(X_test)
                    
                    #code for calculating accuracy 
                    _accuracy_ = accuracy_score(y_test, y_test_predicted, normalize=True)
                    accuracy_list[k_index] = _accuracy_
                    
                    #code for calculating recall 
                    _recalls_ = recall_score(y_test, y_test_predicted, average=None)
                    recall_list[k_index] = _recalls_
                    
                    #code for calculating precision 
                    _precisions_ = precision_score(y_test, y_test_predicted, average=None)
                    precision_list[k_index] = _precisions_
                    
                    #code for calculating confusion matrix 
                    _confusion_matrix_ = confusion_matrix(y_test, y_test_predicted)
                    TN_list[k_index] = _confusion_matrix_[0][0]
                    FP_list[k_index] = _confusion_matrix_[0][1]
                    FN_list[k_index] = _confusion_matrix_[1][0]
                    TP_list[k_index] = _confusion_matrix_[1][1]
                
                # creating a metrics dictionary
                metrics_results['Accuracy'] = np.mean(accuracy_list)
                metrics_results['Precisions'] = np.mean(precision_list, axis=0)
                metrics_results['Recalls'] = np.mean(recall_list, axis=0)
                metrics_results['TN'] = np.mean(TN_list)
                metrics_results['FP'] = np.mean(FP_list)
                metrics_results['FN'] = np.mean(FN_list)
                metrics_results['TP'] = np.mean(TP_list)
                
                if verbose:
                    print('\t\t\tAccuracy:', metrics_results['Accuracy'])
                    print('\t\t\tPrecision:', metrics_results['Precisions'])
                    print('\t\t\tRecall:', metrics_results['Recalls'])
                
                year_results[str(df_index+1)+'year'] = metrics_results   
                
            imputer_results[imputer_name] = year_results
            
        model_results[model_name] = imputer_results  
        
    return model_results


results = perform_data_modeling(models_dictionary, imputed_oversampled_dataframes_dictionary, verbose=True, k_folds=5)


# model -> imputer -> year
def perform_model_ranking(models, imputers, results):
    column_headers = ['-'] + list(imputers.keys())
    rows = []
    for model_name, model_details in results.items():
        row = [model_name]
        for imputer_name, imputer_details in model_details.items():
            mean_accuracy = 0
            for year, metrics in imputer_details.items():
                mean_accuracy += metrics['Accuracy']
            mean_accuracy = mean_accuracy/len(imputer_details)
            row.append(mean_accuracy)
        rows.append(row)
    results_df = pd.DataFrame(data=rows, columns = column_headers)
    return results_df


perform_model_ranking(models_dictionary, imputed_oversampled_dataframes_dictionary, results)


# This list stores results of Balanced Bagging classifier obtained by running it for 
# various values of number of estimators in the range of 1 to 30
results_by_estimators = []
for i in range(29):
    models_dictionary['Balanced Bagging'] = BalancedBaggingClassifier(base_estimator = RandomForestClassifier(criterion='entropy'), n_estimators = i+1, bootstrap = True)
    results = perform_data_modeling(models_dictionary, imputed_oversampled_dataframes_dictionary, verbose=True, k_folds=5)
    results_by_estimators.append(results) 


year1_values = []
year2_values = []
year3_values = []
year4_values = []
year5_values = []

# extract corresponding Balanced bagging with Mean imputation
# classification metrics 
def extract_actual_values_from_dict(curr_dict):
    temp_dict = curr_dict['Balanced Bagging']
    return temp_dict['Mean']

for i in range(29):
    curr_dict = results_by_estimators[i]
    curr_result = extract_actual_values_from_dict(curr_dict)
    
        
    year_1_result = curr_result['1year']
    year_2_result = curr_result['2year']
    year_3_result = curr_result['3year']
    year_4_result = curr_result['4year']
    year_5_result = curr_result['5year']
    year1_values.append(year_1_result['Accuracy'])
    year2_values.append(year_2_result['Accuracy'])
    year3_values.append(year_3_result['Accuracy'])
    year4_values.append(year_4_result['Accuracy'])
    year5_values.append(year_5_result['Accuracy'])


import matplotlib.pyplot as plt

estimators = [i+1 for i in range(29)] 

# plot year1, year2, year3, year4 and year5 accuracy values
# for range of estimator values from 1 to 30
plt.plot(estimators, year1_values, '.b-')
plt.plot(estimators, year2_values, '.r-')
plt.plot(estimators, year3_values, '.y-')
plt.plot(estimators, year4_values, '.g-')
plt.plot(estimators, year5_values, '.m-') 
plt.xlabel("\nNumber of estimators")
plt.ylabel("Accuracy")
plt.title("\nEffect of varying number of estimators on the accuracy scores on different datasets\n")

# display legend
plt.plot(10, 0.93, '.b-', label='Year 1')
plt.plot(10, 0.93, '.r-', label='Year 2')
plt.plot(10, 0.93, '.y-', label='Year 3')
plt.plot(10, 0.93, '.g-', label='Year 4')
plt.plot(10, 0.93, '.m-', label='Year 5')

plt.legend(loc='lower right')



