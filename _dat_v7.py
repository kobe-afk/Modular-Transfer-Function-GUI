#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#IMPORTING ALL PACKAGES AND DEFINING GLOBAL VARIABLES

import os        
import csv
import pathlib
import pandas as pd
from datetime import datetime
import re
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()
from scipy.interpolate import CubicSpline   #IMPORTING CUBIC SPLINE 
import itertools

# cwd = os.getcwd()
# print(cwd)

############################################ INITIALIZING MAIN FUNCTIONS ############################################

#  MAIN VARIABLES  
# data       -> LIST WITH FULL DAT FILE
# FFL_values -> LIST CONTAINING ALL FFL VALUES IN ORDER 
# spl_data   -> LIST WITH ALL DAT SEPERATED BY MEASUREMENT INSIDE
# new_data   -> LIST WITH MEASUREMENT DATA FROM SELECTED R,C
# req_rc     -> STRING WITH REQUIRED RC 


class dat:
    
    final_df = pd.DataFrame()
    main_header = pd.DataFrame()
    rc = []
    ffl_df =pd.DataFrame()
    camPosDf = pd.DataFrame()
    freqList = []
    
    def __init__(self , FILENAME_DAT):
        self.filename = FILENAME_DAT
        self.main_header= dat.main(self.filename)
    
    @staticmethod
    def loadData(filename):

        # SCANNING THROUGH ALL LINES IN THE CHOSEN FILE
        with open(filename) as f:
            data = f.readlines()           

        # SPLITTING DATA BY DELIMITER '\n' // SPL_DATA[1] == MEASUREMENT 1, SPL_DATA[2] == MEASUREMENT 2 etc...
        spl_data = [list(y) for x, y in itertools.groupby(data, lambda z: z == '\n') if not x]

        return(data,spl_data)

    def datTable(spl_data):
        tab = []
        rc_list = []
        temp = []
        temp_header_vals_list = []
        flag = 0
        test = pd.DataFrame()


        temp_len = len(pd.DataFrame(spl_data[1])[0].str.rsplit(" ",1, expand=True))
        
        default = pd.DataFrame(spl_data[1])[0].str.rsplit(" ",1, expand=True)
        header_cols = default[0].values.tolist()[:-17]


        for i,new_data in tqdm(enumerate(spl_data)):   
            if i == 0:
                temp_PMO = pd.DataFrame(new_data)[0].str.rsplit(" ",1, expand=True)
                PMO = temp_PMO.where(temp_PMO[0].str.contains('Pitch') | temp_PMO[0].str.contains('Max Freq')| temp_PMO[0].str.contains('Optimise AF')).dropna()[1].tolist()

                n_col = int((int(PMO[1]) / int(PMO[0])) +1)     
                h_vals = [str( int(i* ( int(PMO[1]) / (n_col - 1)))) for i in range(n_col)]

                continue

            temp_df = pd.DataFrame(new_data)[0].str.rsplit(" ",1, expand=True)
            temp_header_vals = temp_df[1].values.tolist()[:-17]
            header_cols = temp_df[0].values.tolist()[:-17]


            new_temp_len = len(temp_df)
        #     print(new_temp_len)

            if new_temp_len != temp_len: #and flag == 0:
                header_cols = temp_df[0].values.tolist()[:-17]
                flag = 1


                temp = temp_header_vals[18:20]
                temp_h = header_cols[18:20]

                temp_header_vals.extend(temp)
                header_cols.extend(temp_h)

                del temp_header_vals[18:20]
                del header_cols[18:20]

                temp = temp_header_vals[19:54]
                temp_h = header_cols[19:54]


                del temp_header_vals[19:54]
                del header_cols[19:54]

                temp_header_vals.extend(temp)
                header_cols.extend(temp_h)

            for val in new_data:
                if (val[0] == 'S' or val[0] == 'T') and val[1].isnumeric():
                    tab.append(val)
                    temp_header_vals_list.append(temp_header_vals)
        #     if i == 5:break

        df_header = pd.DataFrame(temp_header_vals_list)
        # header_cols.append('Reason of Fail TAN CAM')
        df_header.columns = header_cols

        # df = pd.DataFrame(tab)[0].str.rsplit(' ', expand = True).drop([1,18], axis = 1)

        #         # UNCOMMENT FOR MMULTIPLE TFC'S
        #         df = pd.DataFrame(tab)[0].str.rsplit(' ', expand = True).iloc[:,:len(h_vals )+ 2]
        #         df.columns = ['Camera Pos',PMO[2][:-1]] + h_vals
        #         dfFinalCols = ['Camera Pos'] + freqs
        #         df = df[dfFinalCols]

        df = pd.DataFrame(tab)[0].str.rsplit(' ', expand = True).iloc[:,:2]
        df.columns = ['Camera Pos',PMO[2][:-1]]
        df = df[['Camera Pos',PMO[2][:-1]]]
        camPosDf = pd.concat([df_header['Position'], df], axis = 1).replace(r'\n','', regex=True)

        Final = pd.concat([df,df_header], axis = 1)
        first_column = Final.pop('Position')
        Final.insert(0, 'Position', first_column)
        Final = Final.replace(r'\n','', regex=True) 

        rc_list = []
        new = []
        CameraComponentsCount = len(Final['Camera Pos'].drop_duplicates())
        for i,val in enumerate(Final['Position']):

            if i % CameraComponentsCount == 0:
                temp_list = [rc_list.count(val) + 1] * CameraComponentsCount
                new.extend(temp_list)
                rc_list.append(val)
        #         print(a.final_df.shape[0],len(new))


        dat.rc = rc_list
        dat.ffl_df = Final[['Position','FFL']]

        Final.insert(1,'No. of Scans',new, True)

        Final = Final.replace(r'\n','', regex=True) 
        
        dat.final_df = Final
        dat.camPosDf = camPosDf
        dat.freqList = h_vals

    # MAIN:
    # ARGUMENTS = FILENAME
    # RETURN = APPENDS CLASS VARIABLES 
    @staticmethod
    def main(filename):
        
        data,spl_data = dat.loadData(filename)
        
        file_header = pd.DataFrame(spl_data[0])[0].str.rsplit("=",1, expand=True)
        file_header[1] = file_header[1].str[:-1]
        
        dat.datTable(spl_data)
        
        return(file_header)

    ###################################### INITIALIZING 'GET' FUNCTIONS ############################################
   
    def get_parameter(self,req_param,req_rc):
        val = self.final_df[req_param].where(self.final_df['Position'] == req_rc).dropna().values.tolist()[0]
        return(val)

    # GETTING RC HEADERS:
    # ARGUMENTS = REQUIRED RC FOR LENS
    # RETURN = GETTING THE METADATA FOR THE CHOSEN RC NUMBER
    def rc_header(self,req_rc):
        return(dat.final_df[dat.final_df['Position'] == req_rc])

