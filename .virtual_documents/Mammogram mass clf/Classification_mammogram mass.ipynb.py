import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic("matplotlib", " inline")

import missingno as msno
# Library for performing k-NN and MICE imputations 

from sklearn.impute import KNNImputer
import fancyimpute
# Library to perform Expectation-Maximization (EM) imputation
import impyute as impy
# To perform mean imputation
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
#To perform kFold Cross Validation
from sklearn.model_selection import KFold


df = pd.read_csv('mammographic_masses.data.txt', na_values= '?', names = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
df.head()


df.dtypes


df.columns


df.describe()


df.isnull().sum()


df.loc[(df['age'].isnull()) |
      (df['shape'].isnull()) |
      (df['margin'].isnull())]


df.columns


class Data:
    """ All basic operations performed on data are stored in this class"""
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def drop_missing(self, verbose=False):
        """ Drops missing values and displays the affect of doing so"""
        cleaned = [df.dropna(axis=0, how='any')]
        if verbose:
            print( 'Original length: ', len(self.dataframe), 'Cleaned length :', len(cleaned), '\tMissing data: ', len(self.dataframe) - len(cleaned))
        else:
            return cleaned
    
    def generate_sparsity_matrix(self):
        """ Generates sparsity matrix for visulization of missing areas in attributes"""
        missing_values = self.dataframe.isnull()
        msno.matrix(missing_values)
    
    def generate_heatmap(self):
        """ Generates heatmaps for visualiztion of correlation between variables"""
        missing_values = [self.dataframe.columns[self.dataframe.isnull().any()].tolist()]
        msno.heatmap(self.dataframe[missing_values], figsize=(20,20))
    
    def do_mean_imputation(self):
        """ Performs mean imputation on dataframe passed as argument and returns a dataframe of same length as of the argument"""
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputed_dataframe = (imputer.fit_transform(self.dataframe))
        print(type(imputed_dataframe))
        #imputed_dataframe.columns = self.dataframe.columns
        #return imputed_dataframe
    
    def do_knn_imputation(self):
        """ Performs KNN imputation on dataframe passed as argument and returns a dataframe of same length as of the argument"""
        imputer = KNNImputer(n_neighbors=5).fit_transform(self.dataframe)
        df = pd.DataFrame(data = imputer)
        return df
    
def set_header(df):
    """ Sets column names for all column features and label"""
    df.columns = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity']


Data = Data(df)


df_new = Data.do_knn_imputation()
set_header(df_new)
masses_data.describe()


df_new.isnull().sum()


X = df_new[['age', 'shape', 'margin', 'density']].values


y = df_new['severity'].values

X_names = ['age', 'shape', 'margin', 'density']
X


import numpy
from sklearn.model_selection import train_test_split

numpy.random.seed(1234)

(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.75, random_state=1)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print(len(X_train_scaled))
print(len(X_test_scaled))


from sklearn.tree import DecisionTreeClassifier

clf= DecisionTreeClassifier(random_state=1)

# Train the classifier on the training set
clf.fit(X_train_scaled, y_train)


# Display the tree
from IPython.display import Image  
from io import StringIO
from sklearn import tree
from pydotplus import graph_from_dot_data 

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=X_names)  
graph = graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  


clf.score(X_test, y_test)


from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=1)

cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=10)

cv_scores.mean()


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, random_state=1)
cv_scores = cross_val_score(clf, X_train, y_train, cv=10)

cv_scores.mean()


from sklearn import svm

C = 1.0
svc = svm.SVC(kernel='linear', C=C)


cv_scores = cross_val_score(svc, X_train, y_train, cv=10)

cv_scores.mean()


from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=10)
cv_scores = cross_val_score(clf, X_train, y_train,  cv=10)

cv_scores.mean()


for n in range(1, 50):
    clf = neighbors.KNeighborsClassifier(n_neighbors=n)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=10)
    print (n, cv_scores.mean())


from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_minmax = scaler.fit_transform(X_train)

clf = MultinomialNB()
cv_scores = cross_val_score(clf, X_train, y_train, cv=10)

cv_scores.mean()


C = 1.0
svc = svm.SVC(kernel='rbf', C=C)
cv_scores = cross_val_score(svc, X_train, y_train, cv=10)
cv_scores.mean()


C = 1.0
svc = svm.SVC(kernel='sigmoid', C=C)
cv_scores = cross_val_score(svc, X_train, y_train, cv=10)
cv_scores.mean()


C = 1.0
svc = svm.SVC(kernel='poly', C=C)
cv_scores = cross_val_score(svc, X_train, y_train, cv=10)
cv_scores.mean()


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
cv_scores = cross_val_score(clf,  X_train, y_train, cv=10)
cv_scores.mean()



