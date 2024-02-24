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


#setup color variables for output
blue = "\033[34m"
green = "\033[92m"
yellow = "\033[93m"
reset = "\033[0m" 

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
    workload

    #Adjust for month
    if month >= 6 and month <= 8:
        workload = round(workload * 1.9)
    elif month == 9:
        workload = round(workload * 1.5)
    elif month == 12:
        workload = round(workload * .8)
    
    #Adjust for day
    if day in [1,2,14,15,16,29,30,31]:
        workload = round(workload * 1.2)
    
    #Adjust for hour
    if hour in [22,23,24,0,1,2,3,4,5]:
        workload = round(workload * .1)
    elif hour in [9,10,11,12,13,14,15]:
        workload = round(workload * 1.4)
    elif hour in [6,20,21]:
        workload = round(workload * .5)
    elif hour in [7,18,19]:
        workload = round(workload * .8)
    elif hour == 16:
        workload = round(workload * 1.3)
    elif hour == 17:
        workload = round(workload * 1.1)
    
    #Adjust for day of week
    if weekday in [3,4,5]:
        workload = round(workload * 1.2)
    elif weekday in [1,7]:
        workload = round(workload * .1)
    
    #Write new row to output
    new_row = [year,month,day,hour,weekday,workload]
    new_row_df = pandas.DataFrame([new_row], columns=['Year','Month','Day','Hour','Weekday','Workload'])
    print(blue + 'New Row', year,month,day,hour,weekday,workload, ' ' + reset)
    new_row_df.to_csv(output_path, mode='a', header=False, index=False)

print('Processing Finished')

    