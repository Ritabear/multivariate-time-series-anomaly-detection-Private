import os
import pandas as pd
import numpy as np

##### Description
## load all csv and merge with time
### Input Variable: 
# - path : where the file place
##### Variable
## num :  point to the column name
## because every csv only got time and unique column name
# all_file_name : list of all folder in that directory
# column_nmae : the name to be set for every second column in csv
# content : first read csv to be merge later
# new_content : get the file data ,name is given for every column to merge success
##### Suggestion To Improve
## os.chdir should be outside 

# path : where all the file place
def merge_all_csv(path):
    # to identity which column name should gave
    num = 0
    os.chdir(r"{}".format(path))
    # all_file_name : all the file which need to merge
    # list all the file name
    all_file_name = os.listdir()
    # column_name : to merge all different column in to one
    column_nmae = ["disk_read_operation",
                    "disk_write_operation",
                    "disk_read_bytes",
                    "disk_write_bytes",
                    "network_in",
                    "network_out",
                    "percentage_CPU"]
    content = pd.read_csv(r"Available_Memory_Bytes.csv", names = ["time","memory"])
    
    for file_name in all_file_name[1:]:
        new_content = pd.read_csv(r"{}".format(file_name),names=["time","{}".format(column_nmae[num])])
        content = pd.merge(content, new_content )
        num+=1
    
    return content

##### Description
## change ground truth according model output within accepted range
### Input Variable:
# gt: (np.array) input must be 0,1
# mod_out : (np.array) input must be 0,1 and same length with gt
# interval: (int) distance range accepted for anomaly point
##### Variable
# am : get np.maskarray which does not have 0
# slices : get list of start end of 1 by mode_out

# numpy array type
def change_gt( gt , mod_out , interval ):
    am=np.ma.masked_where(mod_out!=1,mod_out)    
    slices = np.ma.notmasked_contiguous(am)
    for slice in slices:
        # if gt value 1 in range of slice mod_out and not bigger than interval
        if gt[slice.start:slice.stop].any() and (( int(slice.stop) - int(slice.start) ) <= interval):
            gt[slice.start:slice.stop]=1
        else:
            # if bigger than interval , only change a particular to 1
            for i in range(int(slice.start),int(slice.stop),interval):
                if gt[i:i+interval].any():
                    gt[i:i+interval] = 1
    return gt