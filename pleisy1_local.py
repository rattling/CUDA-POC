# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:50:33 2017

@author: JohnArm
"""

def getData():

    dfMaster= pd.read_csv(os.path.join(base_dir, 'master_file.csv'))
    dfDates = pd.read_csv(os.path.join(base_dir, 't_computation_dates.csv'))
    dfParts = pd.read_csv(os.path.join(base_dir, 't_parts_file.csv'))
    #Pandas does not like null values in integer fields so must change transaction_date1 to float
    dfTrans = pd.read_csv(os.path.join(base_dir, 't_transactions_file.csv'),  dtype={'transaction_date1': 'float64'})

    dfSample=dfMaster.merge(dfParts,on='id').merge(dfTrans,on='id')

    #Add columns to hold calculated data
    dfSample = dfSample.reindex( columns = dfSample.columns.tolist()  + ['part_calculation1','part_calculation2',
                    'part_calculation3', 'part_calculation4','part_calculation5'
                    ,'part_calculation6','part_calculation7','part_calculation8',
                    'part_calculation9', 'total_trans'])
    return dfSample

def sum_trans(x,m, n, z):
    total_trans=0
    for i in range(0, m):
        maxDate=n
        for j in range (0, n):
            maxDate = maxDate - math.isnan(x[i, j])  
        z[i]=maxDate
    return z

#@numba.jit(nopython=True) 
def sum_transNB(dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, n, z):
    total_trans=0
    for i in range(0, n):
        z[i]=9 - math.isnan(dt1[i]) -math.isnan(dt2[i])-math.isnan(dt3[i])-math.isnan(dt4[i])
        -math.isnan(dt5[i])-math.isnan(dt6[i])-math.isnan(dt7[i])-math.isnan(dt8[i])-math.isnan(dt9[i])
    return z

def set_field_rnd(n, l1, l2,z):    
    for i in range(1, l1):
        for j in range (1, l2):
            for k in range (0,n-1):
                z[k] = random.random()*j
    return z

#@numba.jit(nopython=True)
def set_field_rndNB(n, l1, l2, z):  
    for i in range(1, l1):
        for j in range (1, l2):
            for k in range (0,n-1):
                z[k] = random.random()*j
    return z

def runCode():
    rowCount=len(df.index)
    print("Running Code")
    #Run sum_trans
    print("Running sum_trans")
    z = np.empty(rowCount, dtype='int64')
    trans_date_col = [col for col in df if col.startswith('transaction_date')]
    n=df[trans_date_col].values.shape[1]
    result1 = sum_trans(df[trans_date_col].values,rowCount, n,z)   
    df['total_trans']= pd.Series(result1, index=df.index, name='result1')
    print("Running set_field_rnd")
    
    #Run set_field_rnd
    z = np.empty(rowCount, dtype='float64')
    df['part_calculation1'] = set_field_rnd(rowCount, 2557,4, z)
    #z = np.empty(rowCount, dtype='float64')
    #df['part_calculation2'] = set_field_rndNB(rowCount, 2557,4, z)
    #z = np.empty(rowCount, dtype='float64')
    #df['part_calculation3'] = set_field_rndNB(rowCount, 2557,4, z)
    print("Finished Updating DataFrame")

def testCode():
    rowCount=len(df.index)
    print(rowCount)
    #Run sum_trans
    print ("Testing Code")
    print("Testing sum_trans")
    print("Results No Numba")
    z = np.empty(rowCount, dtype='int64')
    result = sum_trans(dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9,rowCount, z)
    #print("Results With Numba")
    #z = np.empty(rowCount, dtype='int64')
    #%timeit -n 10 result1 = sum_transNB(dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, rowCount, z)
    #print("Testing set_field_rnd")
    print("Results No Numba")
    #Run set_field_rnd
    z = np.empty(rowCount, dtype='float64')
    #%timeit -n 10 df['part_calculation1'] = set_field_rnd(rowCount, 2557,4, z)
    #print("Results With Numba")
    #z = np.empty(rowCount, dtype='float64')
    #%timeit -n 10 df['part_calculation1'] = set_field_rndNB(rowCount, 2557,4, z)
    #print("Finished Tests")



#START PROGRAM - MAYBE PUT THIS IN A DRIVER FUNCTION FOR A SINGLE CALL WITH PARAMETERS
import pandas as pd
import numpy as np
from numpy import arange
import random
import math
import os
import sys
print(sys.version)
#import numba
#print(numba.__version__)

#Set base_dir to where your inputs files are located
#base_dir = r'/home/ec2-user/inputs'
base_dir = r'C:/Users/JohnArm/Desktop/Pleisey/inputs'
dfSample=getData();
df=pd.concat([dfSample]*1)
rowCount=len(df.index)

runCode();


#testCode();
#df['total_trans']
pd.options.display.float_format = '{:,.0f}'.format



#MISC - MIGHT USE SOME OF THE STUFF BELOW

#df['part_calculation1']
#df['part_calculation2']
#df['part_calculation3']

   #This is just a convenience as need to refer to these date fields later
    #For production code would do something more elegant but focus here  on Numba speedup

