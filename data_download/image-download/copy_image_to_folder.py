import re
import glob
import csv
import pandas as pd
import os
import argparse
import numpy as np

import shutil


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


parent_directory = sorted(glob.glob( 'image/*/*/*/*.jp2'), key=numericalSort)

dest = 'jp2dump_ar_12_13/AR/'

for file_name in parent_directory:
    #print(file_name)
    filename_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}__SDO_AIA_AIA_171', file_name)    
    filename = filename_match.group(0)
    jpg_filename = filename + ".jp2"
    
    dest_filepath = dest + jpg_filename
    if os.path.exists(dest_filepath) == False:
        shutil.copy(file_name, dest)
        #print('yes')
    
    #break
    #full_file_name = os.path.join(src, file_name)
    #if os.path.isfile(full_file_name):
        #shutil.copy(full_file_name, dest)