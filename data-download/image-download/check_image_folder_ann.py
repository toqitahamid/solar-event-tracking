import glob 
import csv
import pandas as pd 
import re

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

'''
directory = glob.glob('AR/*')


for i in directory:
    
    folder_directory = glob.glob(i + '/*.jpg')
    #print(folder_directory)
    if len(folder_directory) == 1:
        print(i)
        data = [[i]]
        with open('folder.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['folder_name'])
            writer.writerows(data)
        csvFile.close()
'''  
'''
csv_file = 'HMI_3520.ann'
data = pd.read_csv(csv_file, header=None, delim_whitespace=True)

cols = ['file', "x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4"]
data.columns = cols
data.head(3)


last_file = data["file"].iloc[-1]
'''

directory = sorted(glob.glob('solar_image_ar_12_13/AR/*'), key=numericalSort)


for i in directory:
    first_match = re.search(r'HMI_\d{4}', i)
    ann_file = first_match.group(0)
    
    ann_file_location = "solar_annotation_ar_12_13/AR/" + ann_file + '.ann'
    data = pd.read_csv(ann_file_location, header=None, delim_whitespace=True)
    
    cols = ['file', "x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4"]
    data.columns = cols
    data.head(3)
    last_file = data["file"].iloc[-1]
    
    
    
    folder_directory = glob.glob(i + '/*.jpg')
    #print(folder_directory)
    folder_size = len(folder_directory)    
    if folder_size < last_file :
        print(i)
        data = [[ann_file]]
        with open('folder_small.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(data)
        csvFile.close()