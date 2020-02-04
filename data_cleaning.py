import pandas as pd

# read in csv file
df_train = pd.read_csv('3c055e822d5b11ea/train.csv')

# drop accident id column bc it is not a feature in our modeling
df_train = df_train.drop(['Accident_ID'], axis = 1)

# replace severity categories with numerical labels
df_train = df_train.replace(["Minor_Damage_And_Injuries", "Significant_Damage_And_Serious_Injuries", "Significant_Damage_And_Fatalities", "Highly_Fatal_And_Damaging"], [1, 2, 3, 4])

# split test and train data
X = df_train.drop(['Severity'], axis=1)
y = df_train['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
