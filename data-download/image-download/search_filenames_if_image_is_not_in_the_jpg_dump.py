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



csv_parent_directory = sorted(glob.glob( 'filenames/image_names_ar_12_13.csv'), key=numericalSort)



for csv_file in csv_parent_directory:
    data = pd.read_csv(csv_file, usecols=['jp2_filename'])
    filename = data.jp2_filename.tolist()
    
    #event_first_match = re.search(r'SPoCA_\d{10}', csv_file)
    #print(event_first_match.group(0))
    #event_name = event_first_match.group(0)
    
    #event_directory = 'solar_folder_maximal_internal_box/' + event_name + '/'
    #if not os.path.isdir(event_directory):
    #    os.makedirs(event_directory)
    
    src = 'jp2dump/AR/171/'
    #src = 'test_jpgdump/'
    for file in filename:
        filename_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}__SDO_AIA_AIA_171', file)    
        filename = filename_match.group(0)
        jpg_filename = filename + ".jp2"
        
        #dest_filepath = dest + jpg_filename
        src_filepath = src + jpg_filename
        #dest_filepath = event_directory + jpg_filename
        if os.path.exists(src_filepath) == False:
            #shutil.copy(src_filepath, event_directory)
            #print(jpg_filename)
            
            data = [[jpg_filename]]
            
            
            with open('image_file_not_found.csv', 'a', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(data)
            csvFile.close()
            
    
        
    