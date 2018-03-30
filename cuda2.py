# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:19:17 2017

@author: JohnArm
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:50:33 2017

@author: JohnArm
"""

def setUp(numba_flag):
    #START PROGRAM - MAYBE PUT THIS IN A DRIVER FUNCTION FOR A SINGLE CALL WITH PARAMETERS
    global pd, np, arange, base_dir, input_dir, output_dir, os, random, math, sys, time, timer, rowCount, df
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
    if numba_flag ==1:
        global numba
        import numba
        print("Using Numba")
        print(numba.__version__)
    
    #Set input_dir to where your inputs files are located
    #base_dir = r'/home/ec2-user/'
    base_dir = r'C:/Users/JohnArm/Desktop/Pleisey/'
    input_dir = base_dir + 'inputs/'
    output_dir = base_dir + 'outputs/'
    
def df2csv(df,fname,sep=','):
  """
    # function is faster than to_csv
    # 7 times faster for numbers if formats are specified, 
    # 2 times faster for strings.
    # Note - be careful. It doesn't add quotes and doesn't check
    # for quotes or separators inside elements
    # We've seen output time going down from 45 min to 6 min 
    # on a simple numeric 4-col dataframe with 45 million rows.
  """
  myformats=[]
  if len(df.columns) <= 0:
    return
  Nd = len(df.columns)
  Nd_1 = Nd - 1
  formats = myformats[:] # take a copy to modify it
  Nf = len(formats)
  # make sure we have formats for all columns
  if Nf < Nd:
    for ii in range(Nf,Nd):
      coltype = df[df.columns[ii]].dtype
      ff = '%s'
      if coltype == np.int64:
        ff = '%d'
      elif coltype == np.float64:
        ff = '%f'
      formats.append(ff)
  fh=open(fname,'w')
  fh.write(','.join(df.columns) + '\n')
  for row in df.itertuples(index=False):
    ss = ''
    for ii in range(Nd):
      ss += formats[ii] % row[ii]
      if ii < Nd_1:
        ss += sep
    fh.write(ss+'\n')
  fh.close()

def genData(s):
    suffix=str(s)
    dfMaster= pd.read_csv(os.path.join(input_dir, 'master_file.csv'))
    dfDates = pd.read_csv(os.path.join(input_dir, 't_computation_dates.csv'))
    dfParts = pd.read_csv(os.path.join(input_dir, 't_parts_file.csv'))
    #Pandas does not like null values in integer fields so must change transaction_date1 to float
    dfTrans = pd.read_csv(os.path.join(input_dir, 't_transactions_file.csv'),  dtype={'transaction_date1': 'float64'})
    
    dfMaster=pd.concat([dfMaster]*s)
    dfMaster.reset_index(inplace=True)
    dfMaster['id']=dfMaster.index
    fileName='master_file_'+str(s)+'.csv'
    dfMaster.to_csv(os.path.join(input_dir, fileName),index=False)
    
    dfDates=pd.concat([dfDates]*s)
    dfDates.reset_index(inplace=True)
    dfDates['id']=dfDates.index
    fileName='t_computation_dates_'+str(s)+'.csv'
    dfDates.to_csv(os.path.join(input_dir, fileName),index=False)
    
    dfParts=pd.concat([dfParts]*s)
    dfParts.reset_index(inplace=True)
    dfParts['id']=dfParts.index
    fileName='t_parts_file_'+str(s)+'.csv'
    dfParts.to_csv(os.path.join(input_dir, fileName),index=False)
    
    dfTrans=pd.concat([dfTrans]*s)
    dfTrans.reset_index(inplace=True)
    dfTrans['id']=dfTrans.index
    fileName='t_transactions_file_'+str(s)+'.csv'
    dfTrans.to_csv(os.path.join(input_dir, fileName),index=False)
  
def importData(s):
    fileName='master_file_' +str(s)+ '.csv'
    dfMaster= pd.read_csv(os.path.join(input_dir,fileName))
    fileName='t_computation_dates_'+str(s)+'.csv'
    dfDates = pd.read_csv(os.path.join(input_dir,fileName))
    fileName='t_parts_file_'+str(s)+'.csv'
    dfParts  = pd.read_csv(os.path.join(input_dir,fileName))
    #Pandas does not like null values in integer fields so must change transaction_date1 to float
    fileName='t_transactions_file_'+str(s)+'.csv'
    dfTrans  = pd.read_csv(os.path.join(input_dir,fileName),  dtype={'transaction_date1': 'float64'})
    df=dfMaster.merge(dfParts,on='id').merge(dfTrans,on='id')
    rowCount=len(df.index)
    pd.options.display.float_format = '{:,.0f}'.format
    return df

def exportData(df):
    df.to_hdf(os.path.join(output_dir, "Calc.csv"),index=False)  
    #df2csv(df,os.path.join(output_dir, "Calc.csv"))

def sum_trans(x,m, n, z):
    total_trans=0
    for i in range(0, m):
        maxDate=n
        for j in range (0, n):
            maxDate = maxDate - math.isnan(x[i, j])  
        z[i]=maxDate
    return z

#@numba.jit(nopython=True) 
def sum_transNB(x,m, n, z):     
    total_trans=0
    for i in range(0, m):
        maxDate=n
        for j in range (0, n):
            maxDate = maxDate - math.isnan(x[i, j])  
        z[i]=maxDate
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

def runCode(s,l1,l2, numba_flag):    
    setUp(numba_flag)
    prog_start = timer()
    print("Starting Program: Numba_Flag ="+ str(numba_flag))
    print("Importing data")
    func_start = timer()
    df=importData(s)
    rowCount=len(df.index)
    print("Running program on " + str(rowCount)+ " records")
    timing=timer()-func_start
    print("Function: importData duration (seconds):" + str(timing))
    print("Running Code")

    #Run sum_trans
    func_start = timer()
    print("Running sum_trans")
    z = np.empty(rowCount, dtype='int64')
    trans_date_col = [col for col in df if col.startswith('transaction_date')]
    n=df[trans_date_col].values.shape[1]
    if numba_flag == 0:
        result1 = sum_trans(df[trans_date_col].values,rowCount, n,z)  
    else:
        result1 = sum_transNB(df[trans_date_col].values,rowCount, n,z)  
    df['total_trans']= pd.Series(result1, index=df.index, name='result1')
    timing=timer()-func_start
    print("Function: sum_trans duration (seconds):" + str(timing))

    print("Running set_field_rnd")    
    #Run set_field_rnd
    func_start = timer()
    z = np.empty(rowCount, dtype='float64')    
    part_num_col = [col for col in df if col.startswith('part_number')]
    n=df[part_num_col].values.shape[1]
    for i in range(0, n ):
        newCol="part_calculation"+str(i+1)
        if numba_flag == 0:
            result2=set_field_rnd(rowCount, l1, l2,df[part_num_col[i]].values, z )
        else:
            result2=set_field_rndNB(rowCount, l1, l2,df[part_num_col[i]].values, z )        
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




 



