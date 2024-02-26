# Matt Herman
# 2/23/24
# All code within is my own work.

#Import Libraries
import numpy as np
import pandas
import os
import sklearn as skl
#import matplotlib 

import datetime as dt

from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.neighbors import KNeighborsRegressor as knr
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as dtc
#from sklearn import model_selection as ms
#from sklearn.metrics import roc_curve
#from matplotlib import pyplot as plt
#from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_iris
#from sklearn.utils import Bunch, shuffle

#setup color variables for output
blue = "\033[34m"
green = "\033[92m"
yellow = "\033[93m"
reset = "\033[0m" 

#Get working dirctory
current_directory = os.getcwd()

#Import Training Data
import_data_csv = 'Sample_Data.csv'
import_data_path = os.path.join(current_directory,import_data_csv)
import_data = pandas.read_csv(import_data_path)

future_date = np.array([2024,4,24])
days_in_prediction = 1 # valid for 1 - 28


#Split Feature & Target
x_features = import_data.iloc[:, 0:5]
#x_features
#x_features.columns
y_target = import_data.iloc[:, -1:]
#y_target

sort_y = y_target['Workload'].sort_values(ascending=True).reset_index(drop=True)
print(sort_y.all)
peak_index = int(0.98 * len(sort_y))
peak_value = sort_y[peak_index]

high_index = int(0.90 * len(sort_y))
high_value = sort_y[high_index]

average_index = int(0.60 * len(sort_y))
average_value = sort_y[average_index]

low_index = int(0.25 * len(sort_y))
low_value = sort_y[low_index]

y_classification_target = pandas.DataFrame(columns=['Workload'])

for index, row in y_target.iterrows():
    value = row['Workload']

    if value >= peak_value:
        new_row = ['Peak']
        y_classification_target.loc[len(y_classification_target)] = new_row 
    elif value >= high_value:
        new_row = ['High']
        y_classification_target.loc[len(y_classification_target)] = new_row    
    elif value >= average_value:
        new_row = ['Average']
        y_classification_target.loc[len(y_classification_target)] = new_row 
    elif value >= low_value:
        new_row = ['Low']
        y_classification_target.loc[len(y_classification_target)] = new_row 
    else:
        new_row = ['Minimal']
        y_classification_target.loc[len(y_classification_target)] = new_row



def Regression_Tree(): 
    regression_tree_parameters = dtr(criterion='squared_error', min_samples_leaf= 1)
    regression_tree = regression_tree_parameters.fit(x_features, y_target)
    #print(tree.export_text(regression_tree))
    #tree(regression_tree, filled=True, feature_names=x_features.columns)  



    future_datetime = dt.datetime(future_date[0],future_date[1],future_date[2])
    future_weekday = dt.datetime.weekday(future_datetime) + 1
    #adjust for 1-7 
    #future_weekday = future_weekday + 1

    #setup data for running 48 hour prediction set against one model
    #test_year = future_date[0]
    #test_month = future_date[1]
    #test_day = future_date[2]
    test_hour = 0
    #check_day = future_date[2]
    check_hour = 0
    one_day = dt.timedelta(days=1)
    prediction_days = dt.timedelta(days=days_in_prediction)
    end_date = future_datetime + prediction_days + one_day
    end_day = int(end_date.day)
    test_day = int(future_datetime.day)
    predictions = pandas.DataFrame(columns=['Year','Month','Day','Hour','Weekday','Predicted Workload'])

    print(green + 'Starting Regression Tree Prediction' + reset)
    while future_datetime.day < end_date.day:
        
        test_row = [future_datetime.year, future_datetime.month, future_datetime.day, test_hour, future_weekday]
        test_df = pandas.DataFrame([test_row], columns=['Year','Month','Day','Hour','Weekday'])
        workload = regression_tree.predict(test_df)
        #new_row = {'Year': future_date[0], 'Month': future_date[1], 'Day': test_day, 'Hour': test_hour, 'Weekday': future_weekday, 'Predicted Workload': workload[0]}
        new_row = [future_datetime.year, future_datetime.month, future_datetime.day, test_hour, future_weekday,  workload[0]]
        predictions.loc[len(predictions)] = new_row
        #predictions = predictions.append(new_row, ignore_index=True)

        test_hour = test_hour + 1

        if test_hour == 24:
            test_hour = 0
            future_datetime = future_datetime + one_day
            future_weekday = dt.datetime.weekday(future_datetime) + 1
            test_day = future_datetime.day

    
    int_columns = ['Year','Month','Day','Hour','Weekday']
    predictions[int_columns] = predictions[int_columns].astype(int)

    print(blue)
    print(predictions)    
    print(reset)
    print(green + 'Regression Tree Predictions Complete')

def KNeighbors_Regressor(KScore): 
    
    #KScore = 3 # For testing, comment out when runnign function

    kn_regressor_score = knr(n_neighbors=KScore)
    kn_regressor = kn_regressor_score.fit(x_features, y_target)

    future_datetime = dt.datetime(future_date[0],future_date[1],future_date[2])
    future_weekday = dt.datetime.weekday(future_datetime) + 1
    #adjust for 1-7 
    #future_weekday = future_weekday + 1

    #setup data for running 48 hour prediction set against one model
    #test_year = future_date[0]
    #test_month = future_date[1]
    #test_day = future_date[2]
    test_hour = 0
    #check_day = future_date[2]
    check_hour = 0
    one_day = dt.timedelta(days=1)
    prediction_days = dt.timedelta(days=days_in_prediction)
    end_date = future_datetime + prediction_days + one_day
    end_day = int(end_date.day)
    test_day = int(future_datetime.day)
    predictions = pandas.DataFrame(columns=['Year','Month','Day','Hour','Weekday','Predicted Workload'])

    print(green + 'Starting KNeighbors with K=', KScore, 'Prediction' + reset)
    while future_datetime.day < end_date.day:
        
        test_row = [future_datetime.year, future_datetime.month, future_datetime.day, test_hour, future_weekday]
        test_df = pandas.DataFrame([test_row], columns=['Year','Month','Day','Hour','Weekday'])
        workload = kn_regressor.predict(test_df)
        #new_row = {'Year': future_date[0], 'Month': future_date[1], 'Day': test_day, 'Hour': test_hour, 'Weekday': future_weekday, 'Predicted Workload': workload[0]}
        new_row = [future_datetime.year, future_datetime.month, future_datetime.day, test_hour, future_weekday,  workload[0]]
        predictions.loc[len(predictions)] = new_row
        #predictions = predictions.append(new_row, ignore_index=True)

        test_hour = test_hour + 1

        if test_hour == 24:
            test_hour = 0
            future_datetime = future_datetime + one_day
            future_weekday = dt.datetime.weekday(future_datetime) + 1
            test_day = future_datetime.day

    
    int_columns = ['Year','Month','Day','Hour','Weekday']
    predictions[int_columns] = predictions[int_columns].astype(int)

    print(blue)
    print(predictions)    
    print(reset)
    print(green + 'Starting KNeighbors with K=', KScore, 'Prediction Complete' + reset)


def Classification_Tree(): 
    classification_tree_parameters = dtc(criterion='entropy', min_samples_leaf= 1)
    classification_tree = classification_tree_parameters.fit(x_features, y_classification_target)
    #print(tree.export_text(classification_tree))
    #tree(regression_tree, filled=True, feature_names=x_features.columns)  



    future_datetime = dt.datetime(future_date[0],future_date[1],future_date[2])
    future_weekday = dt.datetime.weekday(future_datetime) + 1
    #adjust for 1-7 
    #future_weekday = future_weekday + 1

    #setup data for running 48 hour prediction set against one model
    #test_year = future_date[0]
    #test_month = future_date[1]
    #test_day = future_date[2]
    test_hour = 0
    #check_day = future_date[2]
    check_hour = 0
    one_day = dt.timedelta(days=1)
    prediction_days = dt.timedelta(days=days_in_prediction)
    end_date = future_datetime + prediction_days + one_day
    end_day = int(end_date.day)
    test_day = int(future_datetime.day)
    predictions = pandas.DataFrame(columns=['Year','Month','Day','Hour','Weekday','Predicted Workload'])

    print(green + 'Starting Classification Tree Prediction' + reset)
    while future_datetime.day < end_date.day:
        
        test_row = [future_datetime.year, future_datetime.month, future_datetime.day, test_hour, future_weekday]
        test_df = pandas.DataFrame([test_row], columns=['Year','Month','Day','Hour','Weekday'])
        workload = classification_tree.predict(test_df)
        #new_row = {'Year': future_date[0], 'Month': future_date[1], 'Day': test_day, 'Hour': test_hour, 'Weekday': future_weekday, 'Predicted Workload': workload[0]}
        new_row = [future_datetime.year, future_datetime.month, future_datetime.day, test_hour, future_weekday,  workload[0]]
        predictions.loc[len(predictions)] = new_row
        #predictions = predictions.append(new_row, ignore_index=True)

        test_hour = test_hour + 1

        if test_hour == 24:
            test_hour = 0
            future_datetime = future_datetime + one_day
            future_weekday = dt.datetime.weekday(future_datetime) + 1
            test_day = future_datetime.day

    
    int_columns = ['Year','Month','Day','Hour','Weekday']
    predictions[int_columns] = predictions[int_columns].astype(int)

    print(blue)
    print(predictions)    
    print(reset)
    print(green + 'Classification Tree Predictions Complete' + reset)


Regression_Tree()
print()
KNeighbors_Regressor(1)
print()
KNeighbors_Regressor(3)
print()
KNeighbors_Regressor(5)
print()
Classification_Tree()