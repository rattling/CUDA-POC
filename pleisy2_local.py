# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:50:33 2017

@author: JohnArm
"""

def setUp():
    #START PROGRAM - MAYBE PUT THIS IN A DRIVER FUNCTION FOR A SINGLE CALL WITH PARAMETERS
    global pd, np, arange, base_dir, input_dir, output_dir, os, random, math, sys, time, timer, rowCount
    import pandas as pd
    import numpy as np
    from numpy import arange
    import random
    import math
    import os
    import sys
    print(sys.version)
    import time
    from timeit import default_timer as timer
    #import numba
    #print(numba.__version__)
    
    #Set input_dir to where your inputs files are located
    #input_dir = r'/home/ec2-user/inputs'
    base_dir = r'C:/Users/JohnArm/Desktop/Pleisey/'
    input_dir = base_dir + 'inputs/'
    output_dir = base_dir + 'outputs/'

def genData(s):
    suffix=str(s)
    dfMaster= pd.read_csv(os.path.join(input_dir, 'master_file.csv'))
    dfDates = pd.read_csv(os.path.join(input_dir, 't_computation_dates.csv'))
    dfParts = pd.read_csv(os.path.join(input_dir, 't_parts_file.csv'))
    #Pandas does not like null values in integer fields so must change transaction_date1 to float
    dfTrans = pd.read_csv(os.path.join(input_dir, 't_transactions_file.csv'),  dtype={'transaction_date1': 'float64'})
    
    dfMaster=pd.concat([dfMaster]*s)
    fileName='master_file_'+str(s)+'.csv'
    dfMaster.to_csv(os.path.join(input_dir, fileName),index=False)
    
    dfDates=pd.concat([dfDates]*s)
    fileName='t_computation_dates_'+str(s)+'.csv'
    dfDates.to_csv(os.path.join(input_dir, fileName),index=False)
    
    dfParts=pd.concat([dfParts]*s)
    fileName='t_parts_file_'+str(s)+'.csv'
    dfParts.to_csv(os.path.join(input_dir, fileName),index=False)
    
    dfTrans=pd.concat([dfTrans]*s)
    fileName='t_transactions_file_'+str(s)+'.csv'
    dfTrans.to_csv(os.path.join(input_dir, fileName),index=False)
  

def importData(s):
    fileName='master_file_'+str(s)+'.csv'
    dfMaster= pd.read_csv(os.path.join(input_dir, 'master_file.csv'))
    fileName='t_computation_dates_'+str(s)+'.csv'
    dfDates = pd.read_csv(os.path.join(input_dir, 't_computation_dates.csv'))
    fileName='t_parts_file_'+str(s)+'.csv'
    dfParts = pd.read_csv(os.path.join(input_dir, 't_parts_file.csv'))
    #Pandas does not like null values in integer fields so must change transaction_date1 to float
    fileName='t_transactions_file_'+str(s)+'.csv'
    dfTrans = pd.read_csv(os.path.join(input_dir, 't_transactions_file.csv'),  dtype={'transaction_date1': 'float64'})
    df=dfMaster.merge(dfParts,on='id').merge(dfTrans,on='id')
    rowCount=len(df.index)
    pd.options.display.float_format = '{:,.0f}'.format
    return df

def exportData(df):
    df.to_csv(os.path.join(output_dir, "Calc.csv"),index=False)
    
  

def sum_trans(x,m, n, z):
    func_start = timer()
    total_trans=0
    for i in range(0, m):
        maxDate=n
        for j in range (0, n):
            maxDate = maxDate - math.isnan(x[i, j])  
        z[i]=maxDate
    timing=timer()-func_start
    print("Function: sum_trans duration (seconds):" + str(timing))
    return z

#@numba.jit(nopython=True) 
def sum_transNB(x,m, n, z):
    func_start = timer()
    total_trans=0
    for i in range(0, m):
        maxDate=n
        for j in range (0, n):
            maxDate = maxDate - math.isnan(x[i, j])  
        z[i]=maxDate
    timing=timer()-func_start
    print("Function: sum_trans duration (seconds):" + str(timing))
    return z


def set_field_rnd(n, l1, l2,x, z):    
    for i in range(0, l1):
        for j in range (0, l2):
            for k in range (0,n):
                z[k] = random.random()*j + x[k]  
    return z

#@numba.jit(nopython=True)
def set_field_rndNB(n, l1, l2,x, z):    
    for i in range(0, l1):
        for j in range (0, l2):
            for k in range (0,n):
                z[k] = random.random()*j + x[k]
    return z

def runCode(s, numba_flag):    
    prog_start = timer()
    
    print("Importing data")
    func_start = timer()
    df=importData(s)
    rowCount=len(df.index)
    timing=timer()-func_start
    print("Function: importData duration (seconds):" + str(timing))
    print("Running Code")
    
    #Run sum_trans
    print("Running sum_trans")
    z = np.empty(rowCount, dtype='int64')
    trans_date_col = [col for col in df if col.startswith('transaction_date')]
    n=df[trans_date_col].values.shape[1]
    if numba_flag == 0:
        result1 = sum_trans(df[trans_date_col].values,rowCount, n,z)  
    else:
        result1 = sum_transNB(df[trans_date_col].values,rowCount, n,z)  
    df['total_trans']= pd.Series(result1, index=df.index, name='result1')
    print("Running set_field_rnd")
    
    #Run set_field_rnd
    func_start = timer()
    z = np.empty(rowCount, dtype='float64')    
    part_num_col = [col for col in df if col.startswith('part_number')]
    n=df[part_num_col].values.shape[1]
    for i in range(0, n ):
        newCol="part_calculation"+str(i+1)
        if numba_flag == 0:
            result2=set_field_rnd(rowCount, 2557, 4,df[part_num_col[i]].values, z )
        else:
            result2=set_field_rndNB(rowCount, 2557, 4,df[part_num_col[i]].values, z )        
        df[newCol]=pd.Series(result2, index=df.index, name='result2')
    timing=timer()-func_start
    print("Function: set_field_rnd duration (seconds):" + str(timing))
    print("Finished set_field_rnd")
    
    func_start = timer()
    print("Exporting data")
    df=exportData(df)    
    timing=timer()-func_start
    print("Function: exportData duration (seconds):" + str(timing))
    
    timing=timer()-prog_start    
    print("Total Program Duration (seconds):" + str(timing))
    



 



