import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn import cross_validation, preprocessing
from sklearn import tree, neighbors
import warnings
warnings.filterwarnings("ignore")

training_data = pd.read_csv('/scratch/mparker3/train.csv')
test_data = pd.read_csv('/scratch/mparker3/test.csv')


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

dates = []
for i in x_train:
    dates.append(i[0])
x_train_copy = []
for i in range(len(x_train)):
    x_train_copy.append(x_train[i][1:])
x_train_copy = preprocessing.scale(x_train_copy)

for i in range(len(x_train)):
    np.insert(x_train[i],0, dates[i])

X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.40, random_state=1338)


"""for i in x_train:
    i = i[1:]
this lil bit of array slicing will remove the date"""

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

#X_train = cut_date(X_train)
#X_test = cut_date(X_test)
count = 0
X_train_copy = X_train
tests_before_test = []
predicted_values = []
KNN_y_train = []
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
    if len(counts_before_test) >= 5:
        est = neighbors.KNeighborsRegressor(n_neighbors=5)
        est.fit(data_before_test, counts_before_test)
        predicted_values.append(est.predict(i[1:]))
        KNN_y_train.append(counts_before_test[len(counts_before_test)-1])
    count += 1

for i in range(len(predicted_values)):
    if predicted_values[i] < 0:
        predicted_values[i] = 0
print rmsle(KNN_y_train, predicted_values)



"""
# In[ ]:

dtr = tree.DecisionTreeRegressor(max_depth=8)
X_train = cut_date(X_train)
dtr.fit(X_train, Y_train)
X_test = cut_date(X_test)
print X_train, X_test
print dtr.predict(X_test[1])
print Y_test[1]



# In[176]:

for i in range(len(predicted_values)):
    if predicted_values[i] < 0:
        predicted_values[i] = 0
print rmsle(Y_test, predicted_values)


# In[ ]:

def rmsle(y, y_pred):
    for i in range(len(y)):
        if y_pred[i] < 0:
            print "doesn't work"
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

#taken from https://www.kaggle.com/marknagelberg/caterpillar-tube-pricing/rmsle-function
#NOTE: when using this function with the Gradient Boosting regressor, we saw unexpected behavior: namely, the GBR returning negative values when those were not possible in this data set.
#To mitigate this, we've set all negative values returned to 0.


# In[131]:

dtr.fit(X_train, Y_train)
print dtr.predict(X_test)
print len(Y_test)
print rmsle(Y_test, dtr.predict(X_test))


# In[22]:

est_pred = est.predict(X_test)
for i in range(len(est_pred)):
    if est_pred[i] < 0:
        est_pred[i] = 0
print rmsle(Y_test, est_pred)

for i in X_test:
    print i


"""
