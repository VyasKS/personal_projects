import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

# Library for performing k-NN imputations

from sklearn.impute import KNNImputer
import fancyimpute
# Library to perform Expectation-Maximization (EM) imputation
import impyute as impy
# To perform mean imputation
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer

#To perform kFold Cross Validation
from sklearn.model_selection import KFold
path='mammographic_masses.data.txt'

df = pd.read_csv(path, na_values='?', names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
df.head()


