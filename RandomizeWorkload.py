# Matt Herman
# 2/23/24
# All code within is my own work.


#Import Libraries
import numpy as np
import pandas
import os
#import sklearn as skl
#import matplotlib 

from datetime import datetime
#from sklearn import datasets
#from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier as dtc
#from sklearn import model_selection as ms
#from sklearn.metrics import roc_curve
#from matplotlib import pyplot as plt
#from sklearn.metrics import roc_auc_score
#from sklearn.datasets import load_iris
#from sklearn.utils import Bunch, shuffle

current_directory = os.getcwd()

time_span_csv = '2023_By_Hours.csv'
time_span_path = os.path.join(current_directory, time_span_csv)
time_span = pandas.read_csv(time_span_path)

output_csv = 'Sample_Data.csv'
output_path = os.path.join(current_directory, output_csv)

for index, row in time_span.iterrows():
    year = row['Year']
    month = row['Month']
    day = row['Day']
    hour = row['Hour']
    weekday = row['Weekday']

    workload = np.random.randint(50, 101)
    