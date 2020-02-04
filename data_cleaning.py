import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder

# read in csv file
df_train = pd.read_csv('3c055e822d5b11ea/train.csv')

# drop accident id column bc it is not a feature in our modeling
df_train = df_train.drop(['Accident_ID'], axis = 1)

# replace severity categories with numerical labels
df_train = df_train.replace(["Minor_Damage_And_Injuries", "Significant_Damage_And_Serious_Injuries", "Significant_Damage_And_Fatalities", "Highly_Fatal_And_Damaging"], [1, 2, 3, 4])

# split test and train data
X = df_train.drop(['Severity'], axis=1)
y = df_train['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
