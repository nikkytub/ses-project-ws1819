# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 12:36:34 2018

@author: Hanggai
"""

import statsmodels
import csv
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing




##Reading data
with open('Excercise3-data.csv','r', encoding='UTF-8') as f:
    f_tsv = csv.reader(f, delimiter=',')
    for line in f_tsv:
        labels = line
        break
    
attr = []   
for i in range(len(labels)):
    with open('Excercise3-data.csv','r', encoding='UTF-8') as f:
        f_tsv = csv.reader(f, delimiter=',') 
        attr.append([row[i] for row in f_tsv])

##Selecting features: day, hour, month and temperature
day = []
hour = []
trend = []
month = []

i = 0

for item in attr[0]:
    if i == 0:
        i += 1
        continue
    trend.append(float(i))
    i += 1
    day.append(float(item.split(' ')[0][-2:]))
    hour.append(float(item.split(' ')[-1][0:2])+float(item.split(' ')[-1][3:5])/60)
    month.append(float(item.split(' ')[0][5:7]))
  
    
j = 0
TMP = [] #temperature

for item in attr[3]:
    if j == 0:
        j += 1
        continue
    TMP.append(float(item))
    
b1 = []
b2 = []

j = 0
for item in attr[1]:
    if j == 0:
        j += 1
        continue
    b1.append(float(item))
    
j = 0
for item in attr[2]:
    if j == 0:
        j += 1
        continue
    b2.append(float(item))

#b1 = preprocessing.scale(b1)
#b2 = preprocessing.scale(b2)
   
##constructing MLR features
f1 = np.array(trend[0:27742])
f2 = np.multiply(np.array(day[0:27742]), np.array(hour[0:27742]))
f3 = np.array(month[0:27742])
f4 = np.multiply(np.array(month[0:27742]), np.array(TMP[0:27742]))
f5 = np.multiply(np.array(month[0:27742]), np.square(np.array(TMP[0:27742])))
f6 = np.multiply(np.array(month[0:27742]), np.power(np.array(TMP[0:27742]),3))
f7 = np.multiply(np.array(hour[0:27742]), np.array(TMP[0:27742]))
f8 = np.multiply(np.array(hour[0:27742]), np.square(np.array(TMP[0:27742])))
f9 = np.multiply(np.array(hour[0:27742]), np.power(np.array(TMP[0:27742]),3))

#Trainning set
X = np.transpose(np.matrix([f1,f2,f3,f4,f5,f6,f7,f8,f9]))
Y1 = np.array(b1[0:27742])
Y2 = np.array(b2[0:27742])

#X = preprocessing.scale(X)
X = sm.add_constant(X) 

#Traning using model OLS for building 1 and 2 accordingly
model1 = sm.OLS(Y1, X)
result1 = model1.fit()
print(result1.summary())
 
model2 = sm.OLS(Y2, X)
result2 = model2.fit()
print(result2.summary())  

#Predicting data set
p1 = np.array(trend[27742:28078])
p2 = np.multiply(np.array(day[27742:28078]), np.array(hour[27742:28078]))
p3 = np.array(month[27742:28078])
p4 = np.multiply(np.array(month[27742:28078]), np.array(TMP[27742:28078]))
p5 = np.multiply(np.array(month[27742:28078]), np.square(np.array(TMP[27742:28078])))
p6 = np.multiply(np.array(month[27742:28078]), np.power(np.array(TMP[27742:28078]),3))
p7 = np.multiply(np.array(hour[27742:28078]), np.array(TMP[27742:28078]))
p8 = np.multiply(np.array(hour[27742:28078]), np.square(np.array(TMP[27742:28078])))
p9 = np.multiply(np.array(hour[27742:28078]), np.power(np.array(TMP[27742:28078]),3))

Xnew = np.transpose(np.matrix([p1,p2,p3,p4,p5,p6,p7,p8,p9]))

#Xnew = preprocessing.scale(Xnew)
Xnew = sm.add_constant(Xnew,has_constant='add')

#Predict
Ypred1 = result1.predict(Xnew)
Ypred2 = result2.predict(Xnew)

Yact1 = b1[27742:28078]
Yact2 = b2[27742:28078]




#plotting
plt.plot(p1, Ypred1)
plt.plot(p1, Yact1)
plt.xticks([27750,27800,27850,27900,27950,28000,28050],
           ['08-01','08-02','08-03','08-04','08-05','08-06','08-07',])

plt.gca().legend(('prediction','actual'))
#plt.title('Building 1')
plt.xlabel('date')
plt.ylabel('Single Household Load in kW (ANN)')
plt.show()

plt.plot(p1, Ypred2)
plt.plot(p1, Yact2)
plt.xticks([27750,27800,27850,27900,27950,28000,28050],
           ['08-01','08-02','08-03','08-04','08-05','08-06','08-07',])
plt.gca().legend(('prediction','actual'))
#plt.title('Building 2')
plt.xlabel('date')
plt.ylabel('Office Load in kW (ANN)')
plt.show()

'''
import csv
j = 0
Ypred = []
for i in range(len(Ypred2)):
    if i%2 == 0:
        
        Ypred.append(Ypred2[i])
       
with open('Load_TEL.csv', 'w', newline='',encoding='UTF-8') as csv_file:
    csv_writer = csv.writer(csv_file)
   # for b in y_pv_hat[17:41,1]:
        
    csv_writer.writerow(Ypred)  


import csv
Ypred = []
for i in range(len(Ypred1)):
    if i%2 == 0:
        
        Ypred.append(Ypred1[i])
with open('Load_house.csv', 'w', newline='',encoding='UTF-8') as csv_file:
    csv_writer = csv.writer(csv_file)
   # for b in y_pv_hat[17:41,1]:
        
    csv_writer.writerow(Ypred)  
'''
#compute the error
rmse1 = sqrt(mean_squared_error(Yact1, Ypred1))
rmse2 = sqrt(mean_squared_error(Yact2, Ypred2))

nrmse1 = rmse1/np.mean(Yact1) 
nrmse2 = rmse2/np.mean(Yact2)   

#pnorm1 = adjusted_pnorm_error(Yact1, Ypred1, omega=3, p=2)
#pnorm2 = adjusted_pnorm_error(Yact2, Ypred2, omega=3, p=2)

#autocorrelation function of errors
statsmodels.graphics.tsaplots.plot_acf(Yact2 - Ypred2,lags = 100)