# 1 FILE INPUT

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

############################################# IMPORTING CSV DATA #####################################################
class csv:
    
    PMO = []  
    final_df = pd.DataFrame()
    df_csv = pd.DataFrame()
    n_meas = []
    rc = []
    camComp = []
    listSteps = []
    
    def __init__(self ,FILENAME_CSV, dat_df, freqChosen):
        self.filename = FILENAME_CSV
        self.dat_df = dat_df
        
        self.final_df,self.df_csv, self.PMO, self.camComp, self.listSteps= csv.main(self.filename, self.dat_df,freqChosen)
        
    def main(filename, dat_df, freqChosen):

        #IMPORTING THE PITCH MEASUREMENT OPTISE AF DATA USING READ CSV AND READING THE FIRST 3 ROWS
        PMO_csv = pd.read_csv(filename,names = ['Field','value'], nrows = 3)

        PMO = [PMO_csv.iloc[0, 1],PMO_csv.iloc[1, 1],PMO_csv.iloc[2, 1]]
        #         print(PMO)

        with open(filename) as f:
                data = f.readlines()
        
        
        l = ['Field','Val']+[x for x in range(54)]

        PMO_csv = pd.read_csv(filename,names = l, nrows = 3)
        PMO = [PMO_csv.iloc[0, 1],PMO_csv.iloc[1, 1],PMO_csv.iloc[2, 1]]
        n_meas = [ x  for x,val in enumerate(data) if 'Measurement' in val] 

        rc =  [data[i+1].rsplit(' ',1)[1][:-1] + ',' + data[i+2].rsplit(' ',1)[1][:-1] for i, val in enumerate(data) if val == '[Measurement]\n']

        # CSV FIELD NAMES
        n_col = int((int(PMO[1]) / int(PMO[0])) +1)                             # INT(MAX FREQUENCY / PITCH) + 1
        h_vals = [str( int(i* ( PMO[1] / (n_col - 1)))) for i in range(n_col)]  # FOR I = I * MAX FREQ / (COL - 1)
        headers= ["Steps", "Z Distance", "Camera","Core Components"] + h_vals  # APPENDING FIELD NAME LIST

        df_csv = pd.read_csv(filename,skiprows = 3, names = headers,usecols = ["Steps", "Z Distance", "Camera","Core Components", freqChosen])

        cc = df_csv["Core Components"].str[0] + df_csv["Camera"].str.split(" ", expand = True)[1].astype(str)
        df_csv.insert(4, "CC", cc)

        ffl_df = dat_df[['Position','FFL']]
        ffl_df = ffl_df.drop_duplicates()
        list_of_ffl = ffl_df['FFL'].values.tolist()                    # get list of ffl values     

        list_of_z_distance_per_result = []                              # duplicate the z distance for result 11 times for each rc that we have   (may change depending , in this case since the last row has only results and no steps 
                                                                        # we will not be taking the measurements from the row 54 at all)

        list_of_ffl_per_rc = []                                         # duplicate the ffl number 11 times for each rc that we have 

        start_idx = df_csv.index[df_csv['CC'] == 'S1'].tolist()     # find all the index positions of S1 in the column CC


        list_of_steps = df_csv.loc[df_csv['Steps'].str.startswith('S')]['Steps'].drop_duplicates().tolist()
        nSteps = len(list_of_steps)


        column_index = df_csv.index[df_csv['Steps'] == 'Column'].tolist()
        column_index.extend([df_csv.shape[0]])



        s1_t9_labels = df_csv['CC'].drop_duplicates().dropna().tolist()
        ############### gets all the s1 to t9 


        s1_t9_vals = df_csv.loc[(df_csv['Steps'].str.startswith('Step'))][freqChosen].tolist()   # values of s1 to t9 of the whole column '52'
        
        global number_of_splits
        number_of_splits = len(s1_t9_vals)/len(s1_t9_labels)


        split_s1_t9 = np.array_split(s1_t9_vals, number_of_splits)
#         print(split_s1_t9)

        ############# Find the diff between the indexes of the start indexes , this is to find how many steps there are for each rc number   (change to list comprehension)

        list_check_rc_num_placement = [start_idx.index(start_idx[i]) for i in range(len(start_idx)-1) if start_idx[i+1]-start_idx[i] > len(s1_t9_labels)]
        list_check_rc_num_placement.extend([len(start_idx)-1,len(start_idx)])


        list_check_rc_num_placement.insert(0,0)
        steps_in_measurement = [list_check_rc_num_placement[i+1]-list_check_rc_num_placement[i] if i == 0 else ((list_check_rc_num_placement[i+1]-list_check_rc_num_placement[i])-1) for i in range(len(list_check_rc_num_placement)-1)]
        del list_check_rc_num_placement[0]

        ############ getting the steps

        splitting_csv_by_each_Rc = [df_csv.iloc[column_index[n]:column_index[n+1]]for n in tqdm(range(len(column_index)-1))] # split the csv file by each rc


        find_list_of_steps_each_Rc = [i.loc[i['Steps'].str.startswith('S')]['Steps'].drop_duplicates().tolist() for i in tqdm(splitting_csv_by_each_Rc)] # get a list of steps for each rc
        #################### duplicate the R and C values as many times as required (as there are steps per rc number)  

        df_r = df_csv[['Z Distance']].where(df_csv['Steps'] == 'Row').dropna()['Z Distance'].values.tolist()
        df_c = df_csv[['Z Distance']].where(df_csv['Steps'] == 'Column').dropna()['Z Distance'].values.tolist()



        i = 0 # we have to start from -1 if not the final column and row will not be printed out
        store_r_val = []
        store_c_val = []

        for each_idx, rval,cval in tqdm(zip(list_check_rc_num_placement ,df_r,df_c)):
            for x in range(i,each_idx): # creates copies of the rows and columns until the result index
                store_r_val.append(int(rval))
                store_c_val.append(int(cval))

            i = each_idx+1

        ################## Get the z distance


        get_z_dist_from_all_results = df_csv.loc[(df_csv['Steps'] == 'Result') & (df_csv['CC'] == s1_t9_labels[0])]['Z Distance'].tolist()               # gets a list of z distance of all the results in the table sorted by the results and that needs to be taken from the result of any s1 to t9

        z_distance_from_all_steps = df_csv.loc[(df_csv['Steps'].str.startswith('Step')) & (df_csv['CC'] == s1_t9_labels[0])]['Z Distance'].tolist()      # gets a list of z distance from all the steps that are s1


        ######## include a check if measurement_stops_at is the same length as the num_steps_per_rc, if not then we know that there are just some measurements that are results only



        measurement_stops_at = int(len(z_distance_from_all_steps)/len(list_of_steps))  # length of the z_distance_from_all_steps divided by the number of steps 


        if measurement_stops_at != len(get_z_dist_from_all_results):

            correct_ffl_per_rc = list_of_ffl[:measurement_stops_at + 1]    # list up to the rc number

            list_of_steps_in_each_measurement = steps_in_measurement[:measurement_stops_at + 1]

            get_last_correct_z_dist_result = get_z_dist_from_all_results[:measurement_stops_at + 1]    # gets all the correct z_dist result for the measurement that it stops at 

            steps_in_rc = find_list_of_steps_each_Rc[:measurement_stops_at+1]

            ########### loop to duplicate each result and ffl number as many times as there are steps

            for z_dist_res, ffl ,steps in zip(get_last_correct_z_dist_result,correct_ffl_per_rc,list_of_steps_in_each_measurement):   
                list_of_z_distance_per_result += steps * [z_dist_res]
                list_of_ffl_per_rc+= steps * [ffl]
        #     FFL - OFFSET
            list_of_all_z_dist = [(float(ffl_num) - (z_dist_step - z_dist_result)) for z_dist_result, z_dist_step ,ffl_num in zip(list_of_z_distance_per_result,z_distance_from_all_steps,list_of_ffl_per_rc)]   # get the correct z distance formula

        #     # - OFFSET IS TO REVERSE THE X AXIS WITH RESPECT TO THE CSV FILE
        #     list_of_all_z_dist = [(float(ffl_num) - round(z_dist_step - z_dist_result ,3)) for z_dist_result, z_dist_step ,ffl_num in zip(list_of_z_distance_per_result,z_distance_from_all_steps,list_of_ffl_per_rc)]   # get the correct z distance formula

        else:
            steps_in_rc = find_list_of_steps_each_Rc

            for z_dist_res,ffl in zip(get_z_dist_from_all_results,list_of_ffl):   
                list_of_z_distance_per_result += len(list_of_steps) * [z_dist_res]
                list_of_ffl_per_rc+= len(list_of_steps) * [ffl]

        #     print('list_of_ffl_per_rc',list_of_ffl_per_rc)
        #     print('z_dist_step',z_dist_step)
        #     print('z_dist_result',z_dist_result)

            list_of_all_z_dist = [(float(ffl_num) - (z_dist_step - z_dist_result )) for z_dist_result, z_dist_step, ffl_num in zip(list_of_z_distance_per_result,z_distance_from_all_steps,list_of_ffl_per_rc)]

        #     cnt = 0
        #     for z_dist_result, z_dist_step, ffl_num in zip(list_of_z_distance_per_result,z_distance_from_all_steps,list_of_ffl_per_rc):
        #         print(cnt)
        #         cnt = cnt+1
        #         print('FFL NUMBER - ',ffl_num)
        #         print('RESULT - ',z_dist_result)
        #         print('Z DISTANCE - ',z_dist_step)
        #         print()


        #     list_of_all_z_dist = [(float(ffl_num) - round(z_dist_step - z_dist_result ,3)) for z_dist_result, z_dist_step ,ffl_num in zip(list_of_z_distance_per_result,z_distance_from_all_steps,list_of_ffl_per_rc)]
        #     print(list_of_z_distance_per_result,z_distance_from_all_steps,list_of_ffl_per_rc)


        # ############ making of all the parts of the main dataframe 

        s1_t9_only_df = pd.DataFrame(split_s1_t9, columns = s1_t9_labels)
        R = pd.DataFrame(store_r_val)
        C = pd.DataFrame(store_c_val) 
        steps_dataframe = pd.DataFrame(steps_in_rc).stack().reset_index(drop=True)
        z_dist_dataframe = pd.DataFrame(list_of_all_z_dist)


        # ############# making the combined df

        Final = pd.concat([R,C,steps_dataframe,z_dist_dataframe,s1_t9_only_df], axis=1).dropna() # merges all the Dataframes together
        Final.columns = ['R', 'C','steps','Z distance'] + s1_t9_labels
        # Final.columns = ['Z distance'] + s1_t9_labels

        Final["Position"] = Final["R"].astype(str) + ',' +  Final["C"].astype(str)
        first_column = Final.pop('Position')
        Final.insert(0, 'Position', first_column)


        rc_list = []
        new = []
        for i,val in enumerate(Final['Position']):

            if i % nSteps == 0:
                temp_list = [rc_list.count(val) + 1] * nSteps
                new.extend(temp_list)
                rc_list.append(val)

        Final.insert(1,'No. of Scans',new, True)
        Final = Final.drop(['steps'],axis = 1)

    #         print(Final)
        return(Final,df_csv,PMO,s1_t9_labels,list_of_steps)
