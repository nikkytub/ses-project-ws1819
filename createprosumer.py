# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:48:54 2019

@author: Leo77
"""
import random
import numpy as np
from matplotlib import pyplot


'''
#x = [random.gauss(3,1) for _ in range(400)]
#y = [random.gauss(4,2) for _ in range(400)]
a = np.arange(0,23)
bins = a -0.5
#pyplot.hist(a, bins, weights = x, alpha=1, label='x',histtype=u'step')
#pyplot.hist(a, bins, weights = y, alpha=1, label='y',histtype=u'step')

for i in range(0,5):
    pyplot.hist(a, bins, weights = priceAll[i,1:], alpha=1, histtype='step')
    
    

plt.xlabel('hour')
plt.ylabel('electricity price (euro/kwh)')    
pyplot.legend(loc='upper right')
pyplot.show()

for i in range(0,5):
    pyplot.hist(a, bins, weights = co2All[i,1:], alpha=1, histtype='step')
    

plt.xlabel('hour')
plt.ylabel('CO2 emissions (kg CO2/kwh)')      
pyplot.legend(loc='upper right')
pyplot.show()
'''
hour = np.arange(0,23)

for i in range(0,5):
    pyplot.step(hour, priceAll[i,1:], label = 'Prosumer'+str(i+1))
    
    

plt.xlabel('hour')
plt.ylabel('electricity price (euro/kwh)')    
pyplot.legend()
pyplot.show()

for i in range(0,5):
    pyplot.step(hour, co2All[i,1:], label = 'Prosumer'+str(i+1))
    
    

plt.xlabel('hour')
plt.ylabel('CO2 emissions (kg CO2/kwh)')      
pyplot.legend()
pyplot.show()