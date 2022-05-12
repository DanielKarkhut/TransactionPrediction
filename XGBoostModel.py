# %%
# import libraries
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from datetime import timedelta
import math

# sklearn models and evaluators
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# synthetic data
from imblearn.over_sampling import ADASYN

# for graphing
#import plotly.express as px
import datetime  
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Merge both data sets
df = pd.read_csv("transaction_data.csv")

customer_df = pd.read_csv("customer_data.csv")
customer_df = customer_df[['customerId', 'otherLenders', 'isCreditPerformant', 'creditScore', 'businessType']] # Cut down to important features

data = pd.merge(df, customer_df, on='customerId')

# %%
# Features
# customer_id, customerAge, MaritalStatus, educationLevel, location, LoanAmount, interestOnLoan, RepaymentDate, loanRepaidDate, loanStatus, loanCreatedAt, otherLenders, isCreditPerformant, creditScore, businessType

# convert to datetime for formatting
data[['repaymentDate', 'loanRepaidDate', 'loanCreatedAt']] = data[['repaymentDate', 'loanRepaidDate', 'loanCreatedAt']].apply(pd.to_datetime)

# generate days to pay back column
data['paybackDuration'] = (data["loanRepaidDate"] - data["loanCreatedAt"]).dt.days

# create col for the number of days it was overdue (negative numbers means it was paid on time)
data['daysOverdue'] = ((data['repaymentDate'] - data["loanRepaidDate"]) * -1)

# fix NaN values
data[['paybackDuration', 'daysOverdue']] = data[['paybackDuration', 'daysOverdue']].fillna(pd.Timedelta(days=0))

# convert to int64 for label condition
data['interestPercent'] = (data['interestOnLoan'] / data['loanAmount']) * 100

# convert to float
data['daysOverdue'] = (data['daysOverdue'].dt.days).astype(float)

# %%
# Features
# customer_id, customerAge, MaritalStatus, educationLevel, location, LoanAmount, interestOnLoan, RepaymentDate, loanRepaidDate, loanStatus, loanCreatedAt, 
# otherLenders, isCreditPerformant, creditScore, businessType, paybackDuration, daysOverdue, interestPercent
quantitative_features = ['customerAge', 'loanAmount', 'daysOverdue', 'interestPercent', 'creditScore']
qualitative_features = ['MaritalStatus', 'educationLevel', 'location', 'otherLenders', 'businessType', 'isCreditPerformant']
test_features = ['loanStatus']
# drop unused columns
data = data.drop(columns=['customerId', 'repaymentDate', 'loanRepaidDate', 'paybackDuration', 'loanCreatedAt', 'interestOnLoan'])

# save data before fixing skew
original_data = data.copy()
#print(data)
# %%
# create category encoding and set string types to floats
data['loanStatus'] = data['loanStatus'].astype('category')
data['loanStatus'] = data['loanStatus'].cat.codes.astype('int64') # 1: REPAID 2: DELINQUENT

data[['maritalStatus', 'educationLevel', 'otherLenders', 'isCreditPerformant', 'location', 'businessType']] = data[['maritalStatus', 'educationLevel', 'otherLenders', 'isCreditPerformant', 'location', 'businessType']].astype('category')
data['maritalStatus'] = data['maritalStatus'].cat.codes.astype('float64')
data['educationLevel'] = data['educationLevel'].cat.codes.astype('float64')
data['otherLenders'] = data['otherLenders'].cat.codes.astype('float64')
data['isCreditPerformant'] = data['isCreditPerformant'].cat.codes.astype('float64')
data['location'] = data['location'].cat.codes.astype('float64')
data['businessType'] = data['businessType'].cat.codes.astype('float64')

# spearman correlation for checking for redundancy
def correlation_heatmap(data):
	correlations = data.corr()

	fig, ax = plt.subplots(figsize=(10, 10))
	sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
				square=True, linewidths=.5, annot=True, cbar_kws={"shrink":.70})
	plt.show()
correlation_heatmap(data)

# remove the target from the feature set
features = data.iloc[:, data.columns != 'loanStatus']

# clean the dataset of NaN and infinity
def clean_dataset(df):
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
data = pd.merge(clean_dataset(features), data['loanStatus'], left_index=True, right_index=True)

# clean data: remove outliers
data = data[~(data['customerAge'] <= 6)]  
data = data[~(data['daysOverdue'] >= 100)]  
data = data[~(data['interestPercent'] < 2)]  

# Train Test Split
features = data.iloc[:, data.columns != 'loanStatus']
#print(features)
labels = data['loanStatus']
#print(data)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3) # test_size can be 0.2

# %%
# Create Synthetic Data
def makeOverSamplesADASYN(X_train, y_train):
	sm = ADASYN()
	X, y = sm.fit_resample(features, labels)
	return(X,y)

X_train, y_train = makeOverSamplesADASYN(features, labels)

# %%
predictions = pd.DataFrame(y_test)
cfs = {}

# Run XGB Classifier with params
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBClassifier

def xgb_classifier(n_estimators, max_depth, reg_alpha,
                   reg_lambda, min_child_weight, num_boost_round,
                   gamma):
    params = {"booster": 'gbtree',
              "objective" : "binary:logistic",
              "eval_metric" : "auc", 
              "is_unbalance": True,
              "n_estimators": int(n_estimators),
              "max_depth" : int(max_depth),
              "reg_alpha" : reg_alpha,
              "reg_lambda" : reg_lambda,
              "gamma": gamma,
              "num_threads" : 20,
              "min_child_weight" : int(min_child_weight),
              "learning_rate" : 0.01,
              "subsample_freq" : 5,
              "seed" : 42,
              "verbosity" : 0,
              "num_boost_round": int(num_boost_round)}
    train_data = xgb.DMatrix(X_train, y_train)
    cv_result = xgb.cv(params,
                       train_data,
                       1000,
                       early_stopping_rounds=100,
                       stratified=True,
                       nfold=3)
    return cv_result['test-auc-mean'].iloc[-1]

xgbBO = BayesianOptimization(xgb_classifier, {  "n_estimators": (10, 100),
                                                'max_depth': (5, 40),
                                                'reg_alpha': (0.0, 0.1),
                                                'reg_lambda': (0.0, 0.1),
                                                'min_child_weight': (1, 10),
                                                'num_boost_round': (100, 1000),
                                                "gamma": (0, 10)
                                                })

xgbBO.maximize(n_iter=50, init_points=2)

# %%
# Confusion Matrix
params = xgbBO.max['params']
print(params)
params['max_depth']= int(params['max_depth'])
params['n_estimators']= int(params['n_estimators'])

model = XGBClassifier(**params).fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

print(classification_report(y_pred, y_test))

cm = confusion_matrix(y_pred, y_test)
acc = cm.diagonal().sum()/cm.sum()


# add values for confusion matrix visuals
predictions['Optimized XGB'] = y_pred
cfs['Optimized XGB']=(confusion_matrix(y_test, y_pred))

# feature importance
print(model.feature_importances_) # 0 = loanAmount, 1 = interestOnLoan
# plot
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()
print(data.columns)

# %%
# save model 
# model.save_model("daniel_xgb_classifier.joblib")
filename = 'daniel_xgb_classifier..sav'
joblib.dump(model, filename)