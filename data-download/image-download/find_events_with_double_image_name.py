from sunpy.net import hek
import pandas as pd
import os
import csv
import pytz
from datetime import datetime, timedelta
import numpy as np
import math
from sunpy.net.helioviewer import HelioviewerClient
import sunpy.io.jp2 as jp2
import cv2
#from pgmagick import Image
import re
import glob
from shutil import copyfile


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


#filepath = sorted(glob.glob('all_image_csv_12_17/*.csv'), key=numericalSort)
filepath = sorted(glob.glob('../data_collection/grouped_data_12_13_original/AR/*.csv'), key=numericalSort)
#filepath = sorted(glob.glob('grouped_data_all/grouped_data/AR/*.csv'), key=numericalSort)


duplicate_event_list = []


def clean_duplicate_rows(df, event_name):
    #duplicateRowsDF = df[df.duplicated()]
 
    #print("Duplicate Rows except first occurrence based on all columns are :")
    #print(duplicateRowsDF)
    
    df = df.drop_duplicates('event_starttime')
    
    
    df.to_csv('Corrected/' + event_name + '.csv',  index=False)


for i in filepath:
    

    df = pd.read_csv(i)
    event_match = re.search(r'HMI_\d{4}', i)
    event_name = event_match.group(0)
    #print(event_name)
    
    a = df.duplicated(['event_starttime'])
    
    '''
    for j in a:
        if j:
            print(event_name) 
            duplicate_event_list.append(event_name)
            #dst = 'duplicate_events/' + event_name + '.csv'
            #copyfile(i, dst)
            #break
    '''
    if any(a) == True:
        print(event_name)
        #clean_duplicate_rows(df, event_name)


'''
with open('grouped_data_all/duplicate_event_list.txt', 'w') as f:
    for item in duplicate_event_list:
        f.write("%s\n" % item)
'''
