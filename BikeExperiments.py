#CS63 Lab 10
#Matt Parker and Matt Baer
#2017


import numpy as np
import pandas as pd
import math
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.cross_validation import train_test_split
from sklearn import tree, cross_validation, linear_model
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore")

training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


x_data = []
y_data = []
for i in training_data["count"]:
    y_data.append(i)
y_train = np.array(y_data)
for i in range(len(training_data["count"])):
    data = []
    for j in training_data:
        if j!= "count" and j!= "casual" and j!= "registered":
            data.append(training_data[j][i])
    data.append(int(training_data["datetime"][i][11:13]))
    npdata = np.array(data)
    x_data.append(npdata)
x_train = np.array(x_data)


X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1338)

def sort_by_date(train, test):
    data_with_date = zip(train, test)
    data_with_date.sort(key=lambda array: array[0][0])
    return zip(*(data_with_date))[0], zip(*(data_with_date))[1]

X_train, Y_train = sort_by_date(X_train, Y_train)
X_test, Y_test = sort_by_date(X_test, Y_test)

def cut_date(data):
    new_data = []
    for i in range(len(data)):
        new_data.append(data[i][1:])
    return new_data

def rmsle(y, y_pred):
    for i in range(len(y)):
        if y_pred[i] < 0:
            print "doesn't work"
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

    #this function was taken from https://www.kaggle.com/marknagelberg/caterpillar-tube-pricing/rmsle-function


count = 0
tests_before_test = []
predicted_values = []
for i in X_test:
    dates_in_lists = set([])
    counts_before_test = []
    data_before_test = []
    total = str(len(X_test))
    for j in range(len(X_train)):
        if (X_train[j][0] not in dates_in_lists) and (X_train[j][0] < i[0]):
            data_before_test.append(X_train[j])
            dates_in_lists.add(X_train[j][0])
            counts_before_test.append(Y_train[j])

    data_before_test = cut_date(data_before_test)
    print str(count+ 1) + "/" + total

    est = AdaBoostRegressor(GradientBoostingRegressor(max_depth=7))
    est.fit(data_before_test, counts_before_test)
    for val in range(len(i)-1):
         i[val+1] = float(i[val+1])
    predicted_values.append(est.predict(i[1:]))
    count += 1

for i in range(len(predicted_values)):
    if predicted_values[i] < 0:
        predicted_values[i] = 0
        #some regressors gave us negative results, so we set those ones to zero
print "RMSLE: " + rmsle(Y_test, predicted_values)
