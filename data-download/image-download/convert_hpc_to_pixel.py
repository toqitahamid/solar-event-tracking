import re
import glob
import csv
import pandas as pd
import os

import matplotlib.path as mpltPath

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def get_hpc_xy(hpc_bbox):
    Coordinates = hpc_bbox.replace('POLYGON((', '?').replace(" ", '?').replace(",", '?').replace("))", '?').split('?')

    del Coordinates[0]
    del Coordinates[-1]
    
    x = []
    y = []
    
        
    for i in range(0, len(Coordinates), 2):
        x.append(Coordinates[i])
        y.append(Coordinates[i+1])
    return x, y  




def convert_hpc_to_pixel(x, y, cdelt1, cdelt2, crpix1, crpix2):
    pixel_x = []
    pixel_y = []
    if len(x) == 0:
        return None
    for i in range(len(x)):
        pixel_x.append(round(float(crpix1 + (float(x[i]) / cdelt1))))
       
        pixel_y.append(round(float(crpix2 - (float(y[i]) / cdelt2))))
    return pixel_x, pixel_y



def save_max_internal_pixel_coordinate(image_counter, x_min, y_min, x_max, y_max, save_dir, filename):
    data = [[image_counter, x_min, y_max, x_max, y_max, x_max, y_min, x_min, y_min]]
    file_name = str(filename)+'.ann'
    
    save_path = os.path.join(save_dir, file_name)
    
    with open(save_path, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=' ')
        #writer.writerow(['epoch_list', 'train_loss', 'train_loss_mse', 'validation_loss', 'validation_loss_mse','train_time', 'validation_time'])
        writer.writerows(data)
    
    csvFile.close()    



def helioviewer(csv_file, event_type, wavelength):

        data = pd.read_csv(csv_file, usecols=['image_counter', 'bbox', 'cdelt1', 'cdelt2', 'crpix1', 'crpix2', 'filename'])
        
        
        image_counter = data.image_counter.tolist()
        hpc_bbox = data.bbox.tolist()
        cdelt1_list = data.cdelt1.tolist()
        cdelt2_list = data.cdelt2.tolist()
        crpix1_list = data.crpix1.tolist()
        crpix2_list = data.crpix2.tolist()
        #filename = data.filename.tolist()
        
        first_match = re.search(r'HMI_\d{4}', csv_file)
        print(first_match.group(0))
        

        directoryAnnotation = 'solar_annotation/' + event_type + '/' 
        if not os.path.isdir(directoryAnnotation):
            os.makedirs(directoryAnnotation)

        counter = 0

        for item in image_counter:
            bbox_x, bbox_y = get_hpc_xy(hpc_bbox[counter])
            #boundcc_x, boundcc_y = get_hpc_xy(hpc_boundcc[counter])
            cdelt1 = cdelt1_list[counter]
            cdelt2 = cdelt2_list[counter]
            crpix1 = crpix1_list[counter]
            crpix2 = crpix2_list[counter]
            
            #boundcc_pixel_x, boundcc_pixel_y = convert_hpc_to_pixel(boundcc_x, boundcc_y, cdelt1, cdelt2, crpix1, crpix2)
            bbox_pixel_x, bbox_pixel_y = convert_hpc_to_pixel(bbox_x, bbox_y, cdelt1, cdelt2, crpix1, crpix2)
            
            
            
            #final_tl_x, final_tl_y, final_lr_x, final_lr_y = maximal_internal_box(boundcc_pixel_x, boundcc_pixel_y)
            
            annotation_filename = first_match.group(0)
            save_pixel_coordinate(item, bbox_pixel_x, bbox_pixel_y, directoryAnnotation, annotation_filename)
            
            
            #save_max_internal_pixel_coordinate(item, final_tl_x, final_tl_y, final_lr_x, final_lr_y, directoryAnnotation, annotation_filename)
            counter = counter + 1
            
def save_pixel_coordinate(image_counter, pixel_x, pixel_y, save_dir, filename):
    data = [[image_counter, pixel_x[2], pixel_y[2], pixel_x[3], pixel_y[3], pixel_x[0], pixel_y[0], pixel_x[1], pixel_y[1]]]
    file_name = str(filename)+'.ann'
    
    save_path = os.path.join(save_dir, file_name)
    
    with open(save_path, 'a',  newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=' ')
        #writer.writerow(['epoch_list', 'train_loss', 'train_loss_mse', 'validation_loss', 'validation_loss_mse','train_time', 'validation_time'])
        writer.writerows(data)
    
    csvFile.close()             


if __name__ == "__main__":
    
    event_type = 'AR'
    wavelength = '171'
    parent_directory = sorted(glob.glob('event_image_csv_12_13/*.csv'), key=numericalSort)
#    parent_directory = sorted(glob.glob('../data_collection_CH/redownload/' + event_type + '/*.csv'), key=numericalSort)
    
    #df = pd.read_csv('filenames/image_names_ch_12_17.csv', usecols=['event_starttime', 'jp2_filename'])
    
    image_counter = 0
    for csv_file in parent_directory:
        helioviewer(csv_file, event_type, wavelength)
        #print(csv_file)
