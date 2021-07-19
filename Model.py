import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler

from numpy import set_printoptions

from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import chi2

from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline

import pickle

df=pd.read_csv("heart_failure_clinical_records_dataset.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe(include="all"))

plt.figure(figsize=(20, 8))
sns.heatmap(df.corr(), annot=True)
#plt.show()

plt.figure(figsize=(30, 10))
sns.barplot(x=df['age'], y=df['DEATH_EVENT'])
#plt.plot()

sns.countplot(x='DEATH_EVENT',hue='sex',data=df)

df.hist(figsize=(12,8))
#plt.show()

#feature engineering
array = df.values
X = array[:, 0:12]
Y = array[:, 12]
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.20, random_state=1)
'''
#Normalize data
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)

#feature selection
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(normalizedX, Y)

set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(normalizedX)

print(features[0:5,:])



#Data modelling


model = LogisticRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_valid)
print("Score of the model is:")
score = model.score(x_valid, y_valid)
print(score)
print("Accuracy of the model is:")
acc = model.score(x_valid, y_valid)
print(acc*100)

print("Mean absolute error is:")
print(mean_absolute_error(y_valid, predictions))

'''
steps = [('scaler', StandardScaler()),
         ('RFE', RFE(LogisticRegression(), 6)),
         ('lda', LogisticRegression())]

pipeline = Pipeline(steps)
pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_valid)
#print('The accurcay score of the test dataset : ', accuracy_score(y_test, predictions))
#print('\nThe confusion matrix : \n', confusion_matrix(y_test, predictions))
#print('\nFinally the classification report : \n', classification_report(y_test, predictions))
#print('Score : ', pipeline.score(x_test, y_test))

pickle.dump(pipeline,open('Model_1.pkl','wb'))
Model_1=pickle.load(open('Model_1.pkl','rb'))

A = [[53, 1, 90,0,20,1,418000,1.4,139,0,0,43]]
predictions=Model_1.predict(x_valid)
print(predictions)
