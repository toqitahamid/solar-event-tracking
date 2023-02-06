import re
import glob
#USE six for python 2.7 if you want to use urllub.rquest
#from six.moves import urllib
import urllib.request
import urllib.parse
import csv
import json as j
import pandas as pd
import datetime
from datetime import timedelta
from pgmagick import Image
import cv2
import os
import numpy as np
from PIL import Image as PILImage
from urllib.request import urlopen
from requests import get
import time
from shutil import copyfile
from sunpy.net.helioviewer import HelioviewerClient
import sunpy.io.jp2 as jp2
hv = HelioviewerClient()

from maxrect import get_intersection, get_maximal_rectangle, rect2poly
import math

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


def get_header_information(file_name):
    header = jp2.get_header(file_name) 
    head = header[0]
    
    cdelt1 = head['CDELT1']
    cdelt2 = head['CDELT2']
    crpix1 = head['CRPIX1']
    crpix2 = head['CRPIX2']
    
    return cdelt1, cdelt2, crpix1, crpix2


def convert_hpc_to_pixel(x, y, cdelt1, cdelt2, crpix1, crpix2):
    pixel_x = []
    pixel_y = []
    if len(x) == 0:
        return None
    for i in range(len(x)):
        pixel_x.append(round(float(crpix1 + (float(x[i]) / cdelt1))))
       
        pixel_y.append(round(float(crpix2 - (float(y[i]) / cdelt2))))
    return pixel_x, pixel_y



def save_pixel_coordinate(image_counter, x_min, y_min, x_max, y_max, save_dir, filename):
    data = [[image_counter, x_min, y_max, x_max, y_max, x_max, y_min, x_min, y_min]]
    file_name = str(filename)+'.ann'
    
    save_path = os.path.join(save_dir, file_name)
    
    with open(save_path, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=' ')
        #writer.writerow(['epoch_list', 'train_loss', 'train_loss_mse', 'validation_loss', 'validation_loss_mse','train_time', 'validation_time'])
        writer.writerows(data)
    
    csvFile.close()    
    
def save_ann_with_filename(image_counter, bbox, cdelt1, cdelt2, crpix1, crpix2, save_dir, annFilename, filename):
    data = [[image_counter, bbox,  cdelt1, cdelt2, crpix1, crpix2, filename]]
    ann_file_name = str(annFilename)+'.csv'
    
    save_path = os.path.join(save_dir, ann_file_name)
    
    with open(save_path, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        #writer.writerow(['epoch_list', 'train_loss', 'train_loss_mse', 'validation_loss', 'validation_loss_mse','train_time', 'validation_time'])
        writer.writerows(data)
    
    csvFile.close()    
    
def create_ann_with_filename_header(save_dir, annFilename):
    #data = [[image_counter, bbox, boundcc,  cdelt1, cdelt2, crpix1, crpix2, filename]]
    ann_file_name = str(annFilename)+'.csv'
    
    save_path = os.path.join(save_dir, ann_file_name)
    
    with open(save_path, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['image_counter', 'bbox', 'cdelt1', 'cdelt2', 'crpix1', 'crpix2', 'filename'])
        #writer.writerows(data)
    
    csvFile.close()    
        
    
     
def save_events_with_filename(image_counter, filename, save_dir, event_name):
    data = [[image_counter, filename]]
    event_filename = str(event_name)+'.csv'
    
    save_path = os.path.join(save_dir, event_filename)
    
    with open(save_path, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        #writer.writerow(['epoch_list', 'train_loss', 'train_loss_mse', 'validation_loss', 'validation_loss_mse','train_time', 'validation_time'])
        writer.writerows(data)
    
    csvFile.close()    
    
def create_events_with_filename_header(save_dir, event_name):
    #data = [[image_counter, bbox, boundcc,  cdelt1, cdelt2, crpix1, crpix2, filename]]
    event_filename = str(event_name)+'.csv'
    
    save_path = os.path.join(save_dir, event_filename)
    
    with open(save_path, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['image_counter', 'filename'])
        #writer.writerows(data)
    
    csvFile.close()   

def save_gt_coordinate(pixel_x, pixel_y, save_dir, filename):
    data = [[pixel_x[0], pixel_y[0], pixel_x[3], pixel_y[3], pixel_x[2], pixel_y[2], pixel_x[1], pixel_y[1]]]
    file_name = str(filename)+'.ann'
    
    save_path = os.path.join(save_dir, file_name)
    
    with open(save_path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        #writer.writerow(['epoch_list', 'train_loss', 'train_loss_mse', 'validation_loss', 'validation_loss_mse','train_time', 'validation_time'])
        writer.writerows(data)
    
    csvFile.close() 



def helioviewer(csv_file, event_type, wavelength, df, jp2dump_glob):

        data = pd.read_csv(csv_file, usecols=['event_starttime', 'event_endtime', 'hpc_bbox', 'hpc_boundcc', 'event_type'])
        
        
        event_startTime = data.event_starttime.tolist()
        event_endTime = data.event_endtime.tolist()
        hpc_bbox = data.hpc_bbox.tolist()
        hpc_boundcc = data.hpc_boundcc.tolist()
        
        
        
        first_match = re.search(r'HMI_\d{4}', csv_file)
        print(first_match.group(0))
        
        event_filename = first_match.group(0)   
        ann_save_dir = 'event_image_csv_12_13/'
        create_ann_with_filename_header(ann_save_dir, event_filename)
        
        image_save_dir = 'all_image_csv_12_13'
        create_events_with_filename_header(image_save_dir, event_filename)
        
        
        '''  
        #directoryJp2 = 'jp2dump/' + event_type + '/' + first_match.group(0)
        directoryJp2 = 'jp2dump/' + event_type + '/' + wavelength + '/' 
        if not os.path.isdir(directoryJp2):
            os.makedirs(directoryJp2)
                
                
              
        directoryPNG = 'solar_folder_png/' + event_type + '/' + first_match.group(0) + '/'
        if not os.path.isdir(directoryPNG):
            os.makedirs(directoryPNG)
        
        
        directoryJPGdump = 'jpgdump/' + event_type + '/' + wavelength + '/' 
        if not os.path.isdir(directoryJPGdump):
            os.makedirs(directoryJPGdump)
            
            
            
        directoryJPG = 'solar_folder/' + event_type + '/' + first_match.group(0) + '/' 
        if not os.path.isdir(directoryJPG):
            os.makedirs(directoryJPG)
        '''
        #directoryAnnotation = 'solar_annotation/' + event_type + '/' 
        #if not os.path.isdir(directoryAnnotation):
        #    os.makedirs(directoryAnnotation)
                       
                      
        #directoryJpg = input("Enter Jpeg image output directory path: ")
        #directoryJpgCrop = input("Enter cropped Jpeg image with bbox output directory path: ")
        #directoryJpgPoly = input("Enter Jpeg image with polygon output directory path: ")
        counter = 0
        image_counter = 1
        prev_filename = ''
        for event_startTimeData in event_startTime:
            bbox = hpc_bbox[counter]
            boundcc = hpc_boundcc[counter]
            track_idData = counter
            event_endTimeDate = event_endTime[counter]
            '''
            hpc_bboxPolygonData = hpc_bboxPolygon[counter]
           
            Coordinates = hpc_bboxPolygonData.replace('POLYGON((', '?').replace(" ", '?').replace(",", '?').replace(
                "))", '?').split('?')
            x1 = round(float(HPCCENTER + (float(Coordinates[1]) / CDELT)))
            y1 = round(float(HPCCENTER - (float(Coordinates[2]) / CDELT)))
            x2 = round(float(HPCCENTER + (float(Coordinates[3]) / CDELT)))
            y2 = round(float(HPCCENTER - (float(Coordinates[4]) / CDELT)))
            x3 = round(float(HPCCENTER + (float(Coordinates[5]) / CDELT)))
            y3 = round(float(HPCCENTER - (float(Coordinates[6]) / CDELT)))
            x4 = round(float(HPCCENTER + (float(Coordinates[7]) / CDELT)))
            y4 = round(float(HPCCENTER - (float(Coordinates[8]) / CDELT)))
            
            '''
            event_startTimeDataFolder = event_startTimeData
            event_endTimeDateFolder = event_endTimeDate
            '''
            directoryJp2TrackID = directoryJp2 + "/" +  event_startTimeDataFolder.replace(":", "_") + "_" + event_endTimeDateFolder.replace(":", "_") + '/'
            if not os.path.isdir(directoryJp2TrackID):
                os.makedirs(directoryJp2TrackID)
               
                
            directoryPNGTrackID = directoryPNG + "/" + event_startTimeDataFolder.replace(":", "_") + "_" + event_endTimeDateFolder.replace(":", "_") + '/'
            if not os.path.isdir(directoryPNGTrackID):
                os.makedirs(directoryPNGTrackID)    
            
               
            directoryJpgTrackID = directoryJpg + "\\" + event_startTimeDataFolder.replace(":", "_") + "_" + event_endTimeDateFolder.replace(":", "_")
            if not os.path.isdir(directoryJpgTrackID):
                os.makedirs(directoryJpgTrackID)
                
            
            directoryJpgCropTrackID = directoryJpgCrop + "\\" + str(track_idData) + "_" + event_startTimeDataFolder.replace(":",
                                                                                                                    "_") + "_" + event_endTimeDateFolder.replace(
                ":", "_") + "_Count" + str(counter)
            if not os.path.isdir(directoryJpgCropTrackID):
                os.makedirs(directoryJpgCropTrackID)                
               
            '''    
            #Create directory for saving the image with polygon drawing-->start        
            '''
            directoryJpgPolyTrackID = directoryJpgPoly + "\\" + str(
                track_idData) + "_" + event_startTimeDataFolder.replace(":",
                                                                        "_") + "_" + event_endTimeDateFolder.replace(
                ":", "_") + "_Count" + str(counter)
            if not os.path.isdir(directoryJpgPolyTrackID):
                os.makedirs(directoryJpgPolyTrackID)
            '''
            #Create directory for saving the image with polygon drawing-->end    
                
            counter = counter + 1
            
            if counter%20==0:
                #num=counter*5
                #print('Delay after:', str(num))
                time.sleep(5)
            
            start = datetime.datetime.strptime(event_startTimeData, "%Y-%m-%dT%H:%M:%S")
            end = datetime.datetime.strptime(event_endTimeDate, "%Y-%m-%dT%H:%M:%S") - datetime.timedelta(seconds=120)
            
            time_diff = end - start
            time_interval = time_diff/4
            
            annotation_counter = 0
            while start <= end:
                #print("time: " + str(start))
                #hv.download_png('2099/01/01', 4.8, "[SDO,AIA,AIA,304,1,100]", x0=0, y0=0, width=512, height=512)
                    
                
                try:
                    
                    filename = ''
                    #df = pd.read_csv('filenames/image_names_ch_12_17.csv', usecols=['event_starttime', 'jp2_filename'])
                    
                    c = df.index[df['event_starttime'] == str(start)].tolist()
                    #print(c)
                    #print(str(start))
                    
                    if not c:
                        filename = get_filename_from_helioviewer(start)
                        filename_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}__SDO_AIA_AIA_171', filename)
                        
                        filename = filename_match.group(0)
                        
                        filename = filename + ".jp2"
                        
                    else:
                        filename = df.jp2_filename.loc[c[0]]
                        #print(type(filename))
                    
                    #print("filename: " + filename)
                	
                    if (prev_filename != filename): 
                    
                        if (check_if_exist(filename, jp2dump_glob)):
                            #print("hello")
                            jp2_location = get_jp2_file_location(filename, jp2dump_glob)
                            #print(jp2_location)
                            #jpg_location_dst = create_jpg_location(jp2_location, directoryJPG)
                            #print(jpg_location_dst)
                            #jpg_filename = create_jpg_filename(jp2_location)
                            #print(jpg_filename)
                            #if (check_if_exist_jpg(jpg_filename)):
                            #    jpg_location_src = get_jpg_file_location(jpg_filename)
                                #print(jpg_location_src)
                            #    copyfile(jpg_location_src, jpg_location_dst)
                            #else:
                            #    jpg_location_dump = create_jpg_location(jp2_location, directoryJPGdump)
                            
                            #    convert_jp2_to_other_format(jp2_location, jpg_location_dump)
                                
                            #    copyfile(jpg_location_dump, jpg_location_dst)
                            
                            #cdelt1, cdelt2, crpix1, crpix2 = get_header_information(jp2_location)
                            #bbox_pixel_x, bbox_pixel_y = convert_hpc_to_pixel(bbox_x, bbox_y, cdelt1, cdelt2, crpix1, crpix2)
                            #draw_polygon(jp2_location, bbox_pixel_x, bbox_pixel_y, jpg_location)
                            
                            #png_location = create_jpg_location(jp2_location, directoryPNG)
                            #convert_jp2_to_other_format(jp2_location, png_location)
                            
                          
                            #method to save all evenet file names -- here
                            save_image_name_per_event(image_counter, filename, first_match)
                            annotation_counter = create_annotation(annotation_counter, image_counter, jp2_location, bbox, first_match, directoryAnnotation, filename)
                            image_counter += 1
                        
                        '''
                        else:
                            #print('downloading' + str(filename))
                            
                            
                            
                            jp2_location = hv.download_jp2(start, observatory='SDO', instrument='AIA', detector='SPoCA', measurement='193', sourceId = '11', directory=directoryJp2)
                            #print(jp2_location)
                            #if not jp2_location:
                             #   jp2_info = download_jp2(start, directoryJp2, event_type)
                            
                              #  jp2_location = directoryJp2 + jp2_info + '.jp2'    
                            
                            
                            jp2_filename = create_jp2_filename(jp2_location)
                            #print(jp2_filename)
                            #if filename from heliovewer is not matched with the downloaded jp2 image then skip
                            if (filename == jp2_filename):

                                if (os.stat(jp2_location).st_size > 102500):
                                    
                                    jpg_location = create_jpg_location(jp2_location, directoryJPGdump)
                                    
                                    convert_jp2_to_other_format(jp2_location, jpg_location)
                                    
                                    jpg_location_dst = create_jpg_location(jp2_location, directoryJPG)
                                    copyfile(jpg_location, jpg_location_dst)
                                    #print(jpg_location_dst)
                                    
    
                                    #png_location = create_jpg_location(jp2_location, directoryPNGTrackID)
                                    #convert_jp2_to_other_format(jp2_location, png_location)
                                
                                    
                                    annotation_counter = create_annotation(annotation_counter, image_counter, jp2_location, bbox_x, bbox_y, first_match, directoryAnnotation)
                                    image_counter += 1
                                else:
                                    os.remove(jp2_location)
                        '''           
                        
                    start = start + time_interval
                    prev_filename = filename
                    
                     
                        
                except Exception as e:
                    print('Error:', str(e))
                    time.sleep(400)
                    continue


def save_image_name_per_event(image_counter, filename, first_match):
    
    event_name = first_match.group(0)   
    #save_pixel_coordinate(image_counter, x_min, y_min, x_max, y_max, directoryAnnotation, annotation_filename)
    
    save_dir = 'all_image_csv_12_13/'
    save_events_with_filename(image_counter, filename, save_dir, event_name)
    
        
                
                
def create_annotation(annotation_counter, image_counter, jp2_location, bbox, first_match, directoryAnnotation, filename):
    if (annotation_counter == 0):
        cdelt1, cdelt2, crpix1, crpix2 = get_header_information(jp2_location)
        #bbox_pixel_x, bbox_pixel_y = convert_hpc_to_pixel(bbox_x, bbox_y, cdelt1, cdelt2, crpix1, crpix2)
        
        annotation_filename = first_match.group(0)   
        #save_pixel_coordinate(image_counter, x_min, y_min, x_max, y_max, directoryAnnotation, annotation_filename)
        
        save_dir = 'event_image_csv_12_13/'
        save_ann_with_filename(image_counter, bbox,  cdelt1, cdelt2, crpix1, crpix2, save_dir, annotation_filename, filename)
        annotation_counter += 1  
    return annotation_counter              

def merge(list1, list2): 
      
    merged_list = [[list1[i], list2[i]] for i in range(0, len(list1))] 
    return merged_list 


def create_jp2_filename(jp2_location):
    filename_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}__SDO_AIA_AIA_171', jp2_location)    
    filename = filename_match.group(0)
    jp2_filename = filename + ".jp2"
    return jp2_filename
                

def create_jpg_filename(jp2_location):
    filename_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}__SDO_AIA_AIA_171', jp2_location)    
    filename = filename_match.group(0)
    jpg_filename = filename + ".jpg"
    return jpg_filename
   

def create_jpg_location(jp2_location, directoryJPG):
    filename_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}__SDO_AIA_AIA_171', jp2_location)    
    jp2_filename = filename_match.group(0)
    jpg_location = directoryJPG + jp2_filename  + ".jpg"
    return jpg_location


def create_png_location(jp2_location, directoryJPG):
    filename_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}__SDO_AIA_AIA_171', jp2_location)    
    jp2_filename = filename_match.group(0)
    png_location = directoryJPG + jp2_filename  + ".png"
    return png_location
    
def get_filename_from_helioviewer(start):
    dateDownload = str(start.date()) + "T" + str(start.time()) + "Z"
    #print(dateDownload)
    uriHelioviewer = "https://api.helioviewer.org/v2/getJP2Image/?"
    valuesImageUri = {"date": dateDownload, "sourceId": "10", "jpip": "true"}
    dataImageUri = urllib.parse.urlencode(valuesImageUri)
    reqImageUri = urllib.request.Request(uriHelioviewer + dataImageUri)
    respImageUri = urllib.request.urlopen(reqImageUri)
    resImageUriData = respImageUri.read()
    resImageUriData = str(resImageUriData).split("171", 1)[1]
    resImageUriData = resImageUriData.strip("/")
    resImageUriData = resImageUriData.strip("'")
    #print ("Heli Block: " + resImageUriData)
    #resImageUriData = resImageUriData.replace("/", "")
    #resImageUriData = resImageUriData.rstrip("'")
    #resImageUriData = resImageUriData.replace(".jp2'", "")
    #resImageUriData = resImageUriData.strip()
    #print(resImageUriData)
    #urlHelioviewer = "https://api.helioviewer.org/v2/getJP2Image/?"
    #valuesImage = {"date": dateDownload, "sourceId": "11"}
    #dataImage = urllib.parse.urlencode(valuesImage)
    #urllib.request.urlretrieve(urlHelioviewer + dataImage, directoryJp2TrackID + resImageUriData + ".jp2")
    #start = start + datetime.timedelta(seconds=36)
    #x = resImageUriData
        
    return resImageUriData


def get_jp2_file_location(filename, jp2dump_glob):    
    #names = [os.path.basename(x) for x in glob.glob('test/*.jp2')]
    
    for x in jp2dump_glob:
        if (os.path.basename(x) == filename):
            #print(x)
            return x


def check_if_exist(filename, jp2dump_glob):    
    #names = [os.path.basename(x) for x in glob.glob('test/*.jp2')]
    
    for x in jp2dump_glob:
        if (os.path.basename(x) == filename):
            #print(os.path.basename(x))
            return True
        
    return False
    
def get_jpg_file_location(filename):    
    #names = [os.path.basename(x) for x in glob.glob('test/*.jp2')]
    
    for x in glob.glob('jpgdump/AR/171/*.jpg'):
        if (os.path.basename(x) == filename):
            #print(x)
            return x


def check_if_exist_jpg(filename):    
    #names = [os.path.basename(x) for x in glob.glob('test/*.jp2')]
    
    for x in glob.glob('jpgdump/AR/171/*.jpg'):
        if (os.path.basename(x) == filename):
            #print(os.path.basename(x))
            return True
        
    return False
    

def convert_jp2_to_other_format(jp2_img, save_location):
    img = Image(str(jp2_img))
    img.write(str(save_location))

def download_jp2(start, directoryJp2TrackID, event_type):
    sourceId = ''
    wavelength = ''
    
    if (event_type == 'AR'):
        sourceId = '10'
        wavelength = '171'
    elif (event_type == 'CH'):
        sourceId = '11'
        wavelength = '193'
    else:
        print('Event Type Not Correct')
    
    dateDownload = str(start.date()) + "T" + str(start.time()) + "Z"
    #print(dateDownload)
    uriHelioviewer = "https://api.helioviewer.org/v2/getJP2Image/?"
    valuesImageUri = {"date": dateDownload, "sourceId": sourceId, "jpip": "true"}
    dataImageUri = urllib.parse.urlencode(valuesImageUri)
    reqImageUri = urllib.request.Request(uriHelioviewer + dataImageUri)
    respImageUri = urllib.request.urlopen(reqImageUri)
    resImageUriData = respImageUri.read()
    resImageUriData = str(resImageUriData).split(wavelength, 1)[1]
    resImageUriData = resImageUriData.replace(".jp2'", "")
    resImageUriData = resImageUriData.strip()
    #print(resImageUriData)
    urlHelioviewer = "https://api.helioviewer.org/v2/getJP2Image/?"
    valuesImage = {"date": dateDownload, "sourceId": sourceId}
    dataImage = urllib.parse.urlencode(valuesImage)
    urllib.request.urlretrieve(urlHelioviewer + dataImage, directoryJp2TrackID + resImageUriData + ".jp2")
    return resImageUriData

def draw_polygon(image_path, x, y, saved_location):
    im2 = cv2.imread(image_path)
    pts = np.column_stack((x, y))
    #pts = pts[::-1]
    pts = pts.reshape((-1, 1, 2))
    #draw bounding box
    cv2.polylines(im2, [pts], True, (0, 0, 255), 2)
    cv2.imwrite(saved_location, im2)



def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        if response.status_code == 200:
            #print(response)
            # write to file
            file.write(response.content)
        else:
            print('error')



if __name__ == "__main__":
    
    event_type = 'AR'
    wavelength = '171'
    parent_directory = sorted(glob.glob('../data_collection/grouped_data_12_13_original/' + event_type + '/*.csv'), key=numericalSort)
#    parent_directory = sorted(glob.glob('../data_collection_CH/redownload/' + event_type + '/*.csv'), key=numericalSort)
    
    df = pd.read_csv('filenames/image_names_ar_12_13.csv', usecols=['event_starttime', 'jp2_filename'])
    jp2dump_glob = glob.glob('jp2dump/AR/171/*.jp2')
    image_counter = 0
    for csv_file in parent_directory:
        helioviewer(csv_file, event_type, wavelength, df, jp2dump_glob)
        #print(csv_file)
