import glob 
import csv

directory = glob.glob('solar_image_ar_12_13/AR/*')

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
    
    