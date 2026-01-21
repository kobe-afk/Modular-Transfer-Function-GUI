#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import csv
import pathlib
import pandas as pd
from datetime import datetime
import re
import numpy as np
from scipy.interpolate import CubicSpline   #IMPORTING CUBIC SPLINE 
import matplotlib.pyplot as plt   
import math
from tqdm.notebook import tqdm
tqdm.pandas()

# PLOTTING AND GETTING CUBIC SPLINE VALUES V2 // FIRST STEP OF ALGO
def SplineIntersection(x ,y , MTF_Lim,linenum):
    cs = CubicSpline(x, y) # PUTTING CUBIC SPLINE Y PREDICTING FUNCTION IN A VARIABLE
    
    # XS = INTERPOLATED X VALUES INCLUSIVE OF X VALUES
    # xs = np.linspace(x[0],x[-1] , num= len(x) + 3 * (len(x) - 1), endpoint = True) 
    xs = np.linspace(x[0],x[-1] , num= 90+11, endpoint = True)
    
#     MTF_line = np.array([MTF_Lim for i in range(len(xs))])
#     plt.plot(xs, cs(xs), label='data') # PLOTTING THE SPLINE WITH INTERPOLATED VALUES
#     plt.plot(xs, MTF_line, 'r--', color = 'green')
#     plt.xlim(x[0], x[-1]) # SETTING X AXIS LIMIT
#     plt.ylim(0, 100)
    
    
    points_x = [] # LIST TO STORE THE POINTS WHERE YS INTERSECTS WITH MTF LIMIT
    border_val = [] # LIST TO STORE IF THE FIRST OR LAST VALUE IS A BORDER VALUE eg.[True, False]
    sect_result = [] # LIST TO STORE IF THE SECTION PASSED OR FAILED
    
    ys = list(cs(xs))
    for i in range(len(ys)):
        if ys[i]<=0: ys[i] = 0

    flag = (ys[0] >= MTF_Lim) # CHECKS IF THE START VALUE IS ABOVE OR BELOW THE MTF LIMIT
    
    # Z VALUES ARE NEGATIVE
    if( xs[0] <0) & (xs[-1] < 0):

        if flag == True:
            points_x.append(xs[0] * -1)
#             print('UP', xs[i]  * -1)
            border_val.append(True)
        else:
#             points_x.append(xs[0] * 1)
#             print('DOWN',xs[i])
            border_val.append(False)

        for i, val in enumerate(ys):
            nflag = val >= MTF_Lim

            if flag != nflag:
    #             points_x.append(xs[i])
                if nflag: 
                    points_x.append(xs[i] * -1)
#                     print('UP', xs[i]  * -1)
                else:   
                    points_x.append(xs[i] * 1)
#                     print('DOWN',xs[i])

            flag = nflag

        if flag == True:
            points_x.append(xs[i] * -1)
#             print('UP', xs[i]  * -1)
            border_val.append(True)
        else:
            points_x.append(xs[i] * 1)
#             print('DOWN',xs[i])
            border_val.append(False)
        
        n_points = int(len(points_x) / 2) + 1
        temp_pass = []
        cnt = 0
        for i in range(len(points_x)):
            if points_x[i] >= 0:
                if i == len(points_x) - 1:
                    break
                else:
                    cnt = cnt + 1

                    temp_pass.append([linenum, -abs(points_x[i]), -abs(points_x[i+1])])
    
    # Z VALUES ARE POSITIVE        
    else: 
        
        if flag == True:
            points_x.append(xs[0] * 1)
            border_val.append(True)
        else:
            points_x.append(xs[0] * -1)
            border_val.append(False)

        for i, val in enumerate(ys):
            nflag = val >= MTF_Lim

            if flag != nflag:
    #             points_x.append(xs[i])
                if nflag: 
                    points_x.append(xs[i] * 1)
#                     print('UP', xs[i])
                else:   
                    points_x.append(xs[i] * -1)
#                     print('DOWN',xs[i])

            flag = nflag

        if flag == True:
            points_x.append(xs[i] * 1)
            border_val.append(True)
        else:
            points_x.append(xs[i] * -1)
            border_val.append(False)
        
        n_points = int(len(points_x) / 2) + 1
        temp_pass = []
        cnt = 0
        for i in range(len(points_x)):
            if points_x[i] >= 0:
                if i == len(points_x) - 1:
                    break
                else:
                    cnt = cnt + 1

                    temp_pass.append([linenum, points_x[i], abs(points_x[i+1])])
    
    if len(temp_pass) < 1:
        return(-1,-1,-1)
    
    return(temp_pass,points_x,cnt) # RETURNING 

def TFC_Algo(pass_range):

    global tfc_set
    tfc_set = [[-(math.inf), math.inf]]

    for pass_index in range(len(pass_range)):
#         print("Line Num = ", pass_index)

        new_valid_list = []
        for idx in tfc_set:
#             print('IDX = ', idx)
            for new_interval in pass_range[pass_index]:
#                 print('New Interval = ', new_interval)
                
                if new_interval[1] >= idx[1]: continue  #IF LOW OF NEW POINT IS GREATER THAN THE TFC HIGH : CONTINUE
                if new_interval[2] <= idx[0]: continue  #IF HIGH OF NEW POINT IS GREATER THAN THE TFC LOW : CONTINUE
                
                low = max( new_interval[1], idx[0])
                high = min ( new_interval[2], idx[1])
                new_valid_list.append([low, high])
                
        tfc_set = new_valid_list

    return(tfc_set)    

def getTFC(df,MTF_Lim):
    points_f = []
    temp_pass = []
    pass_range = []
    
    final_x = []
    t = []
    
    df_head = df.loc[:, ('Position', 'No. of Scans','R', 'C','Z distance')]
    n_df = df.drop(columns=['Position', 'No. of Scans','R', 'C', 'Z distance'])  # DF WILL BE THE RETURN FROM THE CSV FUNCTION
    
    
    x = df_head['Z distance'].values.tolist()
    new_x = []
    inc_x = 0.0001
    for i in range(len(x)):
        if x[i-1] == x[i]:
            new_x.append(x[i] + inc_x)
            inc_x = inc_x + 0.0001
        else:
            new_x.append(x[i] )


#     line_number = 1
    for i in range(len(n_df.columns.tolist())):
        y = n_df[[x for x in n_df][i]].values
#         print(i)
        pass_ranges,points,cnt = SplineIntersection(new_x, y, MTF_Lim[i], i+1) #FUNC CALL 1
        
        if pass_ranges == -1:
            return('LENS FAIL')
#         print(pass_ranges)
        pass_range.append(pass_ranges)
    
    final = TFC_Algo(pass_range)

    return(final)

def get_interpolated_dfs(list_of_r_c_dfs):
       
    global xs_list
    global ys_list

    xs_list = []
    ys_list = []

    for i in tqdm(range(len(list_of_r_c_dfs))):
        x = list_of_r_c_dfs[i]['Z distance'].values.tolist()
        n_df = list_of_r_c_dfs[i].drop(columns=['Position','No. of Scans','R', 'C', 'Z distance'])
#         rc_num = list_of_r_c_dfs[i].iloc[0, 0:2]       # for testing"
#         print(rc_num)
        

        new_x = []
        inc_x = 0.0001
        for i in range(len(x)):
            if x[i-1] == x[i]:
                new_x.append(x[i] + inc_x)
                inc_x = inc_x + 0.0001
            else:
                new_x.append(x[i])

        for i in range(n_df.shape[1]):
            y = n_df.iloc[:, i].values
#             print(y)

            cs = CubicSpline(new_x, y) # PUTTING CUBIC SPLINE Y PREDICTING FUNCTION IN A VARIABLE  (put y get x )   problem is we cant get y values from the interpolated x points\n",
            #     XS = INTERPOLATED X VALUES INCLUSIVE OF X VALUES\n",

            xs = np.linspace(x[0],x[-1] , num= len(x) + 3 * (len(x) - 1), endpoint = True)
            xs = np.linspace(x[0],x[-1] , num= 90+11, endpoint = True)


            ys = list(cs(xs))
            
            ys_list.append(ys)
            xs_list.append(xs)

    
    
    return(ys_list,xs_list)

def clean_dataframe(df, freq, number_of_interpolated_values, list_of_r_c_dfs,outName):
    
    global list_of_ys_list_df

    global ys_list_df
    
    ############# making a clean dataframe

    ys_list,xs_list = get_interpolated_dfs(list_of_r_c_dfs)

    col_names = df.columns.tolist()[5:]
    
    
    x_interpolated_list = xs_list[::len(col_names)]
    
#     x_interpolated_col_names = [i+"(x)" for i in col_names]
    

    ys_list_df = pd.DataFrame(ys_list).T   # takes the longest
#     xs_list_df = pd.DataFrame(xs_list).T   # takes the longest


    ############ split into the dfs of s1 to t9

    multiples_17 = [n for n in range(0, ys_list_df.shape[1]+len(col_names)) if n % len(col_names) == 0] # CHANGE NAME
    
    list_of_ys_list_df = [ys_list_df.iloc[:,multiples_17[n]:multiples_17[n+1]] for n in (range(len(multiples_17)-1))]  # splitting the temp_df by positions

    renamed_list_of_ys_list_df = []

    for y_df in list_of_ys_list_df:
        y_df.columns = col_names
        renamed_list_of_ys_list_df.append(y_df)


    new_y_df = pd.concat(renamed_list_of_ys_list_df, ignore_index=True)
    new_x_df = pd.DataFrame(x_interpolated_list)
    new_x_df = new_x_df.stack().reset_index(drop=True)
    new_x_df.to_frame()
#     new_x_df.columns = ['Z distance']

    temp_combined_x_y = pd.concat([new_x_df,new_y_df], axis=1)
#     temp_combined_x_y.columns = ['Z distance','S1','S2','T2','S3','T3','S4','T4','S5','T5','S6','T6','S7','T7','S8','T8','S9','T9'] # CHANGE TO NEW FILE FORMAT
    temp_combined_x_y.columns = ['Z distance'] + col_names
    
    ####### adding the rc number to the df

    grouped_df = df.groupby(['Position', 'No. of Scans'], as_index=False, sort=False)
    groupby_rc = [grouped_df.get_group(key).reset_index(drop=True) for key,item in grouped_df]
    
        
    list_of_rcs = [i.iloc[0,0] for i in groupby_rc]
    rc_df = pd.DataFrame(list(np.repeat(list_of_rcs, number_of_interpolated_values)),columns=['Position'])

    list_of_scans = [i.iloc[0,1] for i in groupby_rc]
    sc_df = pd.DataFrame(list(np.repeat(list_of_scans, number_of_interpolated_values)),columns=['Scan No.'])
    
    combined_x_y = pd.concat([rc_df,sc_df,temp_combined_x_y], axis=1)
    
    
    ### export to csv file for combined_x_y
    
    combined_x_y.to_csv(outName, index=False )  

    return(combined_x_y)

def passFailV2(df,MTF_Lim,TFC_Lim):
#     print('Start Pass/Fail')
    df1 = df.iloc[:,3:]
    
    res = df1 - MTF_Lim                                   # SUBTRACTING THE DATAFRAMES
#     print(res)
    res_idx = res.where(res >= 0).dropna().index.tolist() # CHECKING FOR NEGATIVE VALS & STORING INDEX
#     print(len(res_idx))                                          # PRINTING SIZE OF DF

    if res_idx == []:
        return(-1,-1)
    
    passDF = df.loc[res_idx]                                       # MAKING A DATAFRAME BASED ON PREVIOUS INDEX
    temp1 = passDF.groupby(['Position','Scan No.'], as_index=False, sort=False) # GROUPING BY RC
   

    groupPassDF = [temp1.get_group(key) for key,item in temp1]         # MAKING GROUPED OBJECT INTO A LIST

    def has_stepsize_one(it):
        return all(x2 - x1 == 1 for x1, x2 in zip(it[:-1], it[1:]))

    tfc= []
    prev = 0

    for i in tqdm(range(len(groupPassDF))): # LOOPING THROUGH THE LIST OF GROUPED DATAFRAMES
#         print()
#         print('NEXT RC')

        cnt = 0
        flag = 0

        li = groupPassDF[i].index.tolist()
 

        if has_stepsize_one(li) == True:
    #         print('SINGLE TFC', groupPassDF[i].iloc[0,:2].values.tolist())


            if (groupPassDF[i].index.values.tolist()[-1] + 1) % 101 == 0:
    #             print('HIGH POINT', df['Z distance'].iloc[groupPassDF[i].index.values.tolist()[-1]], '//', 'LOW POINT', df['Z distance'].iloc[groupPassDF[i].index.values.tolist()[0]], '//', df['Z distance'].iloc[groupPassDF[i].index.values.tolist()[-1]] - df['Z distance'].iloc[groupPassDF[i].index.values.tolist()[0]]  )
                tfc.append([groupPassDF[i].iloc[0,0], groupPassDF[i].iloc[0,1],df['Z distance'].iloc[groupPassDF[i].index.values.tolist()[-1]] - df['Z distance'].iloc[groupPassDF[i].index.values.tolist()[0]]])
            else:
    #             print('HIGH POINT', df['Z distance'].iloc[groupPassDF[i].index.values.tolist()[-1] + 1], '//', 'LOW POINT', df['Z distance'].iloc[groupPassDF[i].index.values.tolist()[0]], '//', df['Z distance'].iloc[groupPassDF[i].index.values.tolist()[-1] + 1] - df['Z distance'].iloc[groupPassDF[i].index.values.tolist()[0]]  )
                tfc.append([groupPassDF[i].iloc[0,0], groupPassDF[i].iloc[0,1],df['Z distance'].iloc[groupPassDF[i].index.values.tolist()[-1] + 1] - df['Z distance'].iloc[groupPassDF[i].index.values.tolist()[0]]])


        else:
    #         print('MULTI TFC', groupPassDF[i].iloc[0,:2].values.tolist())

            for j, row in groupPassDF[i].iterrows(): # J IS THE INDEX NUMBERS FOR THE DATAFRAME // ROW IS THE VALUE IN IT
                    cnt = cnt+1

                    if j == prev + 1: # IF THE CURRENT INDEX NUMBER IS THE PREV + 1 
                        pass

                    else: # IF THE CURRENT INDEX IS NOT PREV + 1
            #             print(str(int(row[0]))+ ',' + str(int(row[1])))
                        if flag == 0:
                            low = row[2]
                            flag = 1
    #                         print('Low Point','//', j,'//', row[2])

                        else:
                            high = df['Z distance'].iloc[prev + 1]
                            tfc.append([groupPassDF[i].iloc[0,0], groupPassDF[i].iloc[0,1], high - low])

    #                         print('High Point','//', df['Z distance'].iloc[prev + 1], '// tfcList = ', high-low)

                            low = row[2]
                            flag = 0

    #                         print('Low Point','//', j, row[2])



                    if cnt == len(groupPassDF[i]):
                        if (j + 1) % 101 == 0:
                            high = df.loc[j]['Z distance'] 
                        else:
                            high = df.loc[j + 1]['Z distance']

                        tfc.append([groupPassDF[i].iloc[0,0], groupPassDF[i].iloc[0,1], high - low])

    #                     print('High Point','//', j,'//',high, '// tfcList = ', high-low)

        

                    prev = j
            
    finalList = []
    
    for i,val in enumerate(tfc):
        if val[2] >= TFC_Lim:
            finalList.append([val[0],val[1],'1'])
        else:
            finalList.append([val[0],val[1],'0'])
        
#     passVals = [val for val in passList if val[2] == 'Pass']

    
    return(finalList,tfc)

def get_lens_pass_fail(df,freq, number_of_interpolated_values, MTF_Lim,reqTFC, list_of_steps, filename): # PASS IN LIST OF STEPS
    multiples_11 = [n for n in range(0, df.shape[0]+len(list_of_steps)) if n % len(list_of_steps) == 0]


    list_of_r_c_dfs = [df.iloc[multiples_11[n]:multiples_11[n+1]] for n in (range(len(multiples_11)-1))]  # splitting the temp_df by positions
    
    outName = filename + '.csv'
    
    if os.path.isfile(outName):
        print('File exists')
        combined_x_y = pd.read_csv(outName)
        passList,tfcList = passFailV2(combined_x_y,MTF_Lim,reqTFC)
        return(passList,tfcList)
        
    else:
        print('File does not exist')
        combined_x_y = clean_dataframe(df,freq, number_of_interpolated_values,list_of_r_c_dfs,outName) # get_interpolated_dfs will be run inside of clean_dataframe
        passList,tfcList = passFailV2(combined_x_y,MTF_Lim,reqTFC)
        return(passList,tfcList)

def heatMapVals(a,b,c):
    a1 = pd.DataFrame(a)
    b1 = pd.DataFrame(b)
    c1 = pd.DataFrame(c)



    a1[2] = pd.to_numeric(a1[2])
    b1[2] = pd.to_numeric(b1[2])
    c1[2] = pd.to_numeric(c1[2])



    a1['R'] = a1[0].str.split(',',expand = True)[0]
    a1['C'] = a1[0].str.split(',',expand = True)[1]



    b1['R'] = b1[0].str.split(',',expand = True)[0]
    b1['C'] = b1[0].str.split(',',expand = True)[1]



    c1['R'] = c1[0].str.split(',',expand = True)[0]
    c1['C'] = c1[0].str.split(',',expand = True)[1]



    a1['R'] = pd.to_numeric(a1['R'])
    b1['R'] = pd.to_numeric(b1['R'])
    c1['R'] = pd.to_numeric(c1['R'])



    a1['C'] = pd.to_numeric(a1['C'])
    b1['C'] = pd.to_numeric(b1['C'])
    c1['C'] = pd.to_numeric(c1['C'])



    a2 = a1.sort_values(['R', 'C'], ascending=[True, True] , ignore_index = True)
    b2 = b1.sort_values(['R', 'C'], ascending=[True, True] , ignore_index = True)
    c2 = c1.sort_values(['R', 'C'], ascending=[True, True] , ignore_index = True)



    final = a2[0].values.tolist()
    final = pd.DataFrame(final)



    final['Val'] = a2[2] + b2[2] + c2[2]
    final['Val'] = final['Val'] / 3

#     heatMapDF.columns = ['RC','Value']
    
    return(final.values.tolist())
    


# In[24]:


######################################################################################################################
########################################### RUNNING CODE EXAMPLES ####################################################
######################################################################################################################


# # NEW FILE FORMAT
# number_of_interpolated_values = 101

# MTF_Lim = [60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60]
# # MTF_Lim = [10, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# filename = 'Yong Mei_Lens 1 to 15 repeatability_NewMax new batch.csv'

# list_of_steps = ['Step 1','Step 2','Step 3','Step 4','Step 5','Step 6','Step 7','Step 8','Step 9','Step 10','Step 11','Step 12','Step 13','Step 14','Step 15','Step 16','Step 17','Step 18','Step 19','Step 20','Step 21','Step 22','Step 23','Step 24','Step 25']

# PMO_csv = pd.read_csv(filename,names = ['Field','value'], nrows = 3)
# PMO = [PMO_csv.iloc[0, 1],PMO_csv.iloc[1, 1],PMO_csv.iloc[2, 1]]
# freq = PMO[2]

# df_csv = pd.read_csv('final_df_CSVNew_wPOS.csv')
# df_csv = df_csv.drop(columns = 'Unnamed: 0')

# reqTFC = 0.11

# passList,tfcList = get_lens_pass_fail(df_csv,freq, 101, MTF_Lim,reqTFC,list_of_steps,filename)


# In[25]:


# # NEW FILE FORMAT SINGLE TFC
# test = df_csv.where((df_csv['Position'] == '1,2') & (df_csv['No. of Scans'] == 1)).dropna()
# test
# MTF_Lim = [60] * len(test.columns.tolist()[5:])

# final = getTFC(test,MTF_Lim)
# final


# In[26]:


# a = [['1,2', 11, '1'],['1,3', 11, '1'],['1,4', 11, '1'],['1,5', 11, '1'],['1,6', 11, '1'],['1,7', 11, '1'],['1,8', 11, '1'],['1,9', 11, '1'],['1,10', 11, '1'],['2,10', 11, '1'],['2,9', 11, '1'],['2,8', 11, '1'],['2,7', 11, '1'],['2,6', 11, '1'],['1,1', 17, '1']]
# b = [['1,2', 11, '1'],['1,3', 11, '1'],['1,4', 11, '1'],['1,5', 11, '1'],['1,6', 11, '0'],['1,7', 11, '1'],['1,8', 11, '0'],['1,9', 11, '0'],['1,10', 11, '1'],['2,10', 11, '1'],['2,9', 11, '1'],['2,8', 11, '1'],['2,7', 11, '0'],['2,6', 11, '0'],['1,1', 17, '1']]
# c = [['1,2', 11, '1'],['1,3', 11, '1'],['1,4', 11, '1'],['1,5', 11, '0'],['1,6', 11, '0'],['1,7', 11, '1'],['1,8', 11, '0'],['1,9', 11, '0'],['1,10', 11, '1'],['2,10', 11, '1'],['2,9', 11, '1'],['2,8', 11, '0'],['2,7', 11, '0'],['2,6', 11, '0'],['1,1', 17, '1']]

# x = heatMapVals(a,b,c)
# x


# In[ ]:




