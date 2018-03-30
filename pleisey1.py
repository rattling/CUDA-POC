# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 11:33:29 2017

@author: JohnArm
"""
import pandas as pd
import numpy as np
from numpy import arange
import random
import math
import os

base_dir = r'C:/Users/JohnArm/Desktop/Pleisey/inputs'

df = pd.read_csv(os.path.join(base_dir, 'master_file.csv'))
dfDates = pd.read_csv(os.path.join(base_dir, 't_computation_dates.csv'))
dfParts = pd.read_csv(os.path.join(base_dir, 't_parts_file.csv'))
dfTrans = pd.read_csv(os.path.join(base_dir, 't_transactions_file.csv'),  dtype={'transaction_date1': 'float64'})

df=df.merge(dfParts,on='id').merge(dfTrans,on='id')
rowCount=len(df.index)

df = df.reindex( columns = df.columns.tolist()  + ['part_calculation1','part_calculation2',
                'part_calculation3', 'part_calculation4','part_calculation5'
                ,'part_calculation6','part_calculation7','part_calculation8',
                'part_calculation9', 'total_trans'])

def set_field_rnd(n, l1, l2):
    y = np.empty(n, dtype='float64')
    for i in range(1, l1):
        for j in range (1, l2):
            for k in range (0,n-1):
                y[k] = random.random()*j
    return y
    
def sum_trans(dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, n):
    y = np.empty(n, dtype='int64')
    total_trans=0
    for i in range(0, n):
        y[i]=9 - math.isnan(dt1[i]) -math.isnan(dt2[i])-math.isnan(dt3[i])-math.isnan(dt4[i])
        -math.isnan(dt5[i])-math.isnan(dt6[i])-math.isnan(dt7[i])-math.isnan(dt8[i])-math.isnan(dt9[i])
    return y
    
#Note - dont even have to pass in data to function here as its not used - just getting random list back
#Also not sure if using apply is the best? Is it passing in whole vector at once and allowing the parallel
#Could pass in vector of the col and then generate a vector of randoms within numba  
#Can also use the namespace to loop through these but might make it quite hard to read
test=df['part_calculation1'] = set_field_rnd(rowCount, 1,4)
df['part_calculation1'] = set_field_rnd(rowCount, 1,4)
df['part_calculation2'] = set_field_rnd(rowCount, 1,4)
df['part_calculation3'] = set_field_rnd(rowCount, 1,4)
df['part_calculation4'] = set_field_rnd(rowCount, 1,4)
df['part_calculation5'] = set_field_rnd(rowCount, 1,4)
df['part_calculation6'] = set_field_rnd(rowCount, 1,4)
df['part_calculation7'] = set_field_rnd(rowCount, 1,4)
df['part_calculation8'] = set_field_rnd(rowCount, 1,4)
df['part_calculation9'] = set_field_rnd(rowCount, 1,4)

dt1=df['transaction_date1'].values
dt2=df['transaction_date2'].values
dt3=df['transaction_date3'].values
dt4=df['transaction_date4'].values
dt5=df['transaction_date5'].values
dt6=df['transaction_date6'].values
dt7=df['transaction_date7'].values
dt8=df['transaction_date8'].values
dt9=df['transaction_date9'].values

result = sum_trans(dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9,rowCount)
df['total_trans']= pd.Series(result, index=df.index, name='result')

df['part_calculation1'] = set_field_rnd(rowCount, 2557,4)
df['part_calculation2'] = set_field_rnd(rowCount, 2557,4)
df['part_calculation3'] = set_field_rnd(rowCount, 2557,4)
df['part_calculation4'] = set_field_rnd(rowCount, 2557,4)
df['part_calculation5'] = set_field_rnd(rowCount, 2557,4)
df['part_calculation6'] = set_field_rnd(rowCount, 2557,4)
df['part_calculation7'] = set_field_rnd(rowCount, 2557,4)
df['part_calculation8'] = set_field_rnd(rowCount, 2557,4)
df['part_calculation9'] = set_field_rnd(rowCount, 2557,4)

df['part_number1'] = set_field_rnd(rowCount, 2557,4)
df['part_number2'] = set_field_rnd(rowCount, 2557,4)
df['part_number3'] = set_field_rnd(rowCount, 2557,4)
df['part_number4'] = set_field_rnd(rowCount, 2557,4)
df['part_number5'] = set_field_rnd(rowCount, 2557,4)
df['part_number6'] = set_field_rnd(rowCount, 2557,4)
df['part_number7'] = set_field_rnd(rowCount, 2557,4)
df['part_number8'] = set_field_rnd(rowCount, 2557,4)
df['part_number9'] = set_field_rnd(rowCount, 2557,4)
df['part_number10'] = set_field_rnd(rowCount, 2557,4)

#result = sum_trans(df['transaction_date1'].values, df['transaction_date2'].values, df['transaction_date3'].values, rowCount)
#df['total_trans']= pd.Series(result, index=df.index, name='result')





namespace = globals()
for x in range(1, 13):
    namespace['double_%d' % x] = x + 2


