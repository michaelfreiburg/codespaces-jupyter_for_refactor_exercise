#Überarbeitete Version (refactoring)
#actual main code
import yaml

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score, classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve

import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

import yaml
with open ("run_config.yml") as file:
    config = yaml.safe_load(file)

df = pd.read_csv(config["url"])

correlation_matrix = df.corr(numeric_only = True)
print(correlation_matrix)

#drop this column because its data leakage
X = df.iloc[:,:-1].drop(columns = ["duration"])
X.columns

y = df.y

# Now lets split the dataset for training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
# Logistic Regression Model
smtnc = SMOTENC(categorical_features=np.where(X.dtypes == "object")[0]) 
X_train_smote, y_train_smote = smtnc.fit_resample(X_train, y_train)

X_train_smote = pd.DataFrame(X_train_smote,columns = X_train.columns)


categorical_features = X.iloc[:, list(np.where(X.dtypes == "object")[0])].columns
continious_features = X.iloc[:, list(np.where(X.dtypes == "int")[0])].columns
categorical_features, continious_features



numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers = [("num", numeric_transformer, continious_features),
                                                 ("cat", categorical_transformer, categorical_features)])


# Next step is to create a linear regression
l_regr = LogisticRegression(max_iter=1000)
# Next step is to build the pipeline
pipeline_l_regr = Pipeline([("col_transformer", preprocessor), ("estimator", l_regr)])


# Lets run the model to see:
pipeline_l_regr.fit(X_train_smote, y_train_smote)
pred_no_resample = pipeline_l_regr.predict(X_test)

# Now lets evaluate our model

#1. Confusion Matrix
cf = confusion_matrix(y_test, pred_no_resample)
sns.heatmap(cf, annot=True, fmt=".2f")

#2. make classification report
print(classification_report(y_test, pred_no_resample))

# 'lets do class weights'


l_regr2 = LogisticRegression(class_weight="balanced", max_iter = 1000)
pipeline_l_regr2 = Pipeline([("col_transformer", preprocessor),
                             ("estimator", l_regr2)])

pipeline_l_regr2.fit(X_train, y_train)
pred_no_resample2 = pipeline_l_regr2.predict(X_test)

cf = confusion_matrix(y_test, pred_no_resample2)
sns.heatmap(cf, annot=True, fmt='.2f')

print(classification_report(y_test, pred_no_resample2))

# 'now lets use cros_val'
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score

pipeline_l_regr.fit(X_train_smote, y_train_smote)
pred_no_resample = pipeline_l_regr.predict(X_test)


print(cross_val_score(pipeline_l_regr, X_train, y_train, cv=5))



