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

def create_csv(name):
    
    file_name = str(name)+'.csv'
    save_dir = 'csv'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    
    with open(save_path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['event_starttime', 'event_endtime', 'hpc_bbox', 'hpc_boundcc', 'hpc_coord', 'hpc_radius','boundbox_c1ll', 'boundbox_c2ll', 'boundbox_c1ur', 'boundbox_c2ur',  'frm_name', 'frm_specificid', 'event_type', 'obs_channelid'])
    csvFile.close()
    
    
def create_csv_noaa(name):
    
    file_name = str(name)+'.csv'
    save_dir = 'csv'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    
    with open(save_path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['event_starttime', 'event_endtime', 'hpc_bbox', 'hpc_boundcc', 'hpc_coord', 'hpc_radius', 'boundbox_c1ll', 'boundbox_c2ll', 'boundbox_c1ur', 'boundbox_c2ur',  'frm_name', 'ar_noaanum', 'event_type' 'obs_channelid'])
    csvFile.close()    


def save_event_information_noaa(event_starttime, event_endtime, hpc_bbox, hpc_boundcc, hpc_coord, hpc_radius, boundbox_c1ll, boundbox_c2ll, boundbox_c1ur, boundbox_c2ur,  frm_name, ar_noaanum, event_type, obs_channelid, name):
    data = [[event_starttime, event_endtime, hpc_bbox, hpc_boundcc, hpc_coord, hpc_radius, boundbox_c1ll, boundbox_c2ll, boundbox_c1ur, boundbox_c2ur,  frm_name, ar_noaanum, event_type, obs_channelid]]
    file_name = str(name)+'.csv'
    save_dir = 'csv'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    
    with open(save_path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        #writer.writerow(['epoch_list', 'train_loss', 'train_loss_mse', 'validation_loss', 'validation_loss_mse','train_time', 'validation_time'])
        writer.writerows(data)
    
    csvFile.close()

def save_event_information(event_starttime, event_endtime, hpc_bbox, hpc_boundcc, hpc_coord, hpc_radius, boundbox_c1ll, boundbox_c2ll, boundbox_c1ur, boundbox_c2ur,  frm_name, frm_specificid, event_type, obs_channelid, name):
    data = [[event_starttime, event_endtime, hpc_bbox, hpc_boundcc, hpc_coord, hpc_radius, boundbox_c1ll, boundbox_c2ll, boundbox_c1ur, boundbox_c2ur,  frm_name, frm_specificid, event_type, obs_channelid]]
    file_name = str(name)+'.csv'
    save_dir = 'csv'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    
    with open(save_path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        #writer.writerow(['epoch_list', 'train_loss', 'train_loss_mse', 'validation_loss', 'validation_loss_mse','train_time', 'validation_time'])
        writer.writerows(data)
    
    csvFile.close()


def convert_est_to_utc(event_time):
    utc = pytz.utc
    fmt = '%Y-%m-%dT%H:%M:%S'
    eastern=pytz.timezone('US/Eastern')
    date=datetime.datetime.strptime(event_time,"%Y-%m-%dT%H:%M:%S") 
    
    date_eastern = eastern.localize(date,is_dst=None)
    date_utc = date_eastern.astimezone(utc)
    
    converted_event_time = date_utc.strftime(fmt) 
    return converted_event_time

client = hek.HEKClient()
event_type = 'CH'
module = 'SPoCA'
file_starttime = '2012-01-01'
file_endtime = '2013-12-31'



startTime = '2012-01-01 00:02:00'
endTime = '2013-12-31 23:58:00'
#endTime = '2014-12-18 18:26:48'


#results = client.search(hek.attrs.Time(startTime, endTime), hek.attrs.EventType('AR'), hek.attrs.FRM.Name == 'SPoCA', hek.attrs.FRM.SpecificID == 'SPoCA_v1.0_AR_0000021612')
results = client.search(hek.attrs.Time(startTime, endTime), hek.attrs.EventType('CH'), hek.attrs.FRM.Name == 'SPoCA')
#results = client.search(hek.attrs.Time(startTime, endTime), hek.attrs.EventType(event_type), hek.attrs.FRM.Name == 'HMI SHARP')
#results = client.search(hek.attrs.Time(startTime, endTime), hek.attrs.EventType('AR'), hek.attrs.FRM.Name == 'HMI SHARP', hek.attrs.FRM.SpecificID == '5374')
#results = client.search(hek.attrs.Time(startTime, endTime), hek.attrs.EventType('AR'), hek.attrs.FRM.Name == 'NOAA SWPC Observer', hek.attrs.AR.NOAANum ==  '12553')
#results = client.search(hek.attrs.Time(startTime, endTime), hek.attrs.EventType('SG'), hek.attrs.FRM.Name == 'Sigmoid Sniffer', hek.attrs.AR.NOAANum ==  '1998', hek.attrs.OBS.ChannelID == '131_THIN')

# = client.search(hek.attrs.Time(startTime, endTime), hek.attrs.EventType('CH'), hek.attrs.FRM.Name == 'SPoCA', hek.attrs.FRM.SpecificID == 'SPoCA_v1.0_CH_0000023848')
#results = client.search(hek.attrs.Time(startTime, endTime), hek.attrs.EventType('FL'), hek.attrs.FRM.Name == 'SSW Latest Events', hek.attrs.AR.NOAANum ==  '2002')
#results = client.search(hek.attrs.Time(startTime, endTime), hek.attrs.EventType('FL'), hek.attrs.FRM.Name == 'SSW Latest Events')
#SSW_Latest_Events
#SWPC 12002
#[elem["frm_specificid"] for elem in results]
#name = 'FL' + '_' + 'SWPC'+ '_' + '2016-01-01' + '_' + '2016-12-31'

name = event_type + '_' + module + '_' + file_starttime + '_' + file_endtime
create_csv(name)
#create_csv_noaa(name)


excluded_name = 'exluded_' + event_type + '_' + module + '_' + file_starttime + '_' + file_endtime
create_csv(excluded_name)

save_all_name = 'dump_' + event_type + '_' + module + '_' + file_starttime + '_' + file_endtime
create_csv(save_all_name)

for res in results:
    
    
    event_starttime = res['event_starttime']
    #event_starttime = convert_est_to_utc(event_starttime)
    event_endtime = res['event_endtime']
    #event_endtime = convert_est_to_utc(event_endtime)
    hpc_bbox = res['hpc_bbox']
    hpc_boundcc = res['hpc_boundcc']
    hpc_coord = res['hpc_coord']
    hpc_radius = res['hpc_radius']
    boundbox_c1ll = res['boundbox_c1ll']
    boundbox_c2ll = res['boundbox_c2ll']
    boundbox_c1ur = res['boundbox_c1ur']
    boundbox_c2ur = res['boundbox_c2ur']
    frm_name = res['frm_name']
    frm_specificid = res['frm_specificid']
    #ar_noaanum = res['ar_noaanum']
    event_type = res['event_type']
    obs_channelid = res['obs_channelid']
    #name = 'SPoCA_22494'
    save_event_information(event_starttime, event_endtime, hpc_bbox, hpc_boundcc, hpc_coord, hpc_radius, boundbox_c1ll, boundbox_c2ll, boundbox_c1ur, boundbox_c2ur,  frm_name, frm_specificid, event_type, obs_channelid, save_all_name)
    
    
    #for NOAA SWPC Observer
    #save_event_information_noaa(event_starttime, event_endtime, hpc_bbox, hpc_boundcc, hpc_coord, hpc_radius, boundbox_c1ll, boundbox_c2ll, boundbox_c1ur, boundbox_c2ur,  frm_name, ar_noaanum, obs_channelid, save_all_name)
  
'''
limb_x = []
limb_y = []

for angleInDegrees in range(360):
    
    radius = 965.01612
    origin_X = 0.0
    origin_Y = 0.0
    #angleInDegrees = 27
    limb_x.append(float(radius * math.cos(angleInDegrees * math.pi / 180)) + origin_X)
    limb_y.append(float(radius * math.sin(angleInDegrees * math.pi / 180)) + origin_Y)

#(x-center_x)^2 + (y - center_y)^2 < radius^2

center_x = 0.0
center_y = 0.0
#radius = 965.01612
'''

def get_hpc_limb(radius): 
    limb_x = []
    limb_y = []
    
    for angleInDegrees in range(360):
        origin_X = 0.0
        origin_Y = 0.0
        #angleInDegrees = 27
        limb_x.append(float(radius * math.cos(angleInDegrees * math.pi / 180)) + origin_X)
        limb_y.append(float(radius * math.sin(angleInDegrees * math.pi / 180)) + origin_Y)
    return limb_x, limb_y
     

def in_circle(center_x, center_y, radius, x, y):
    square_dist = (center_x - x) ** 2 + (center_y - y) ** 2
    return square_dist <= radius ** 2

#in_circle(center_x, center_y, radius, 784.233, -336.513)


def get_radius(event_starttime):
    
    hv = HelioviewerClient()
    #hv.download_png('2099/01/01', 4.8, "[SDO,AIA,AIA,304,1,100]", x0=0, y0=0, width=512, height=512)
    filepath = hv.download_jp2(event_starttime, observatory='SDO', instrument='AIA', detector='SPoCA', measurement='193', sourceId = '11', directory='helioviewer/CH/')
    radius = get_radius_from_jp2(filepath)
    
    return radius, filepath


def get_radius_from_jp2(filepath):
    jp2_header = jp2.get_header(filepath) 
    jp2_head = jp2_header[0]
    radius = jp2_head['RSUN_OBS']
    #print (radius)
    return radius


def get_header_information(file_name):
    header = jp2.get_header(file_name) 
    head = header[0]
    
    cdelt1 = head['CDELT1']
    cdelt2 = head['CDELT2']
    crpix1 = head['CRPIX1']
    crpix2 = head['CRPIX2']
    
    return cdelt1, cdelt2, crpix1, crpix2


def draw_polygon(image_path, x, y, saved_location):
    im2 = cv2.imread(image_path)
    pts = np.column_stack((x, y))
    #pts = pts[::-1]
    pts = pts.reshape((-1, 1, 2))
    #draw bounding box
    cv2.polylines(im2, [pts], True, (0, 0, 255), 2)
    cv2.imwrite(saved_location, im2)
    
    
def convert_hpc_to_pixel(x, y, cdelt1, cdelt2, crpix1, crpix2):
    pixel_x = []
    pixel_y = []
    if len(x) == 0:
        return None
    for i in range(len(x)):
        pixel_x.append(round(float(crpix1 + (float(x[i]) / cdelt1))))
       
        pixel_y.append(round(float(crpix2 - (float(y[i]) / cdelt2))))
    return pixel_x, pixel_y

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

def convert_jp2_to_png(jp2_img, save_location):
    img = Image(jp2_img)
    img.write(save_location) 



def draw_limb(filepath, radius):
    save_directory = 'helioviewer/CH/jpg/'
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)
        
    bbox_directory = 'helioviewer/CH/bbox/'
    if not os.path.exists(bbox_directory):
        os.makedirs(bbox_directory)        
        
    first_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}', filepath)    
    save_location = save_directory + first_match.group(0) + '.jpg'
    bbox_location = bbox_directory + first_match.group(0) + '.jpg'
    
    cdelt1, cdelt2, crpix1, crpix2 = get_header_information(filepath)
    
    limb_x, limb_y = get_hpc_limb(radius)
    limb_pixel_x, limb_pixel_y = convert_hpc_to_pixel(limb_x, limb_y, cdelt1, cdelt2, crpix1, crpix2)
    
    convert_jp2_to_png(filepath, save_location)
    image = save_location         
    draw_polygon(image, limb_pixel_x, limb_pixel_y, bbox_location)
    
#previous_time = '2015-01-01T00:02:00'

def compare_time(previous_time, start_time):
    present = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S").date()
    past = datetime.strptime(previous_time, "%Y-%m-%dT%H:%M:%S").date()
    return past < present

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def Perimeter(x, y):
    w1 = abs(x[1] - x[0])
    w2 = abs(x[2] - x[3])
    l1 = abs(y[3] - y[0])
    l2 = abs(y[2] - y[1])
    
    perimeter = w1 + w2 + l1 + l2
    return perimeter

def Thinness(area, perimeter):
    thinness = (4 * math.pi) * (area/(perimeter * perimeter))
    #thinness = area/perimeter
    return thinness


#past_radious = 0.0
radius = 0
previous_time = '2011-12-31T23:59:59'
for res in results:
    
    #print(radius)
    start_time = res['event_starttime']
    #print('start_time: ' + start_time )
    if (compare_time(previous_time, start_time)):
        previous_time = start_time
        #print('previous_time: ' + previous_time )
        #download image and get radius ('RSUN_OBS') from the downloaded image
        radius, filepath = get_radius(start_time)
        #past_radious = radius
        
        #draw limb on downloaded image
        #sdraw_limb(filepath, radius)
    
    
    
    bbox = res['hpc_bbox']
    
    x, y = get_hpc_xy(bbox)
    
    x = [float(i) for i in x]
    y = [float(i) for i in y]
    
    #area = PolyArea(x, y)    
    #perimeter = Perimeter(x, y)
    #thinness = Thinness(area, perimeter)
    
    center_x = 0.0
    center_y = 0.0
    
    if (in_circle(center_x, center_y, radius, np.abs(float(x[0])), np.abs(float(y[0]))) 
    and in_circle(center_x, center_y, radius, np.abs(float(x[1])), np.abs(float(y[1])))
    and in_circle(center_x, center_y, radius, np.abs(float(x[2])), np.abs(float(y[2])))
    and in_circle(center_x, center_y, radius, np.abs(float(x[3])), np.abs(float(y[3])))):
        
        
    #if (in_circle(center_x, center_y, radius, np.abs(res['hpc_x']), np.abs(res['hpc_y']))):       
#   if (-725.0 <= np.abs(res['hpc_x']) <= 725.0 and -500.0 <= np.abs(res['hpc_y']) <= 500.0 ):
        event_starttime = res['event_starttime']
        #event_starttime = convert_est_to_utc(event_starttime)
        event_endtime = res['event_endtime']
        #event_endtime = convert_est_to_utc(event_endtime)
        hpc_bbox = res['hpc_bbox']
        hpc_boundcc = res['hpc_boundcc']
        hpc_coord = res['hpc_coord']
        hpc_radius = res['hpc_radius']
        boundbox_c1ll = res['boundbox_c1ll']
        boundbox_c2ll = res['boundbox_c2ll']
        boundbox_c1ur = res['boundbox_c1ur']
        boundbox_c2ur = res['boundbox_c2ur']
        frm_name = res['frm_name']
        frm_specificid = res['frm_specificid']
        #ar_noaanum = res['ar_noaanum']
        event_type = res['event_type']
        obs_channelid = res['obs_channelid']
        #name = 'SPoCA_22494'
        save_event_information(event_starttime, event_endtime, hpc_bbox, hpc_boundcc, hpc_coord, hpc_radius, boundbox_c1ll, boundbox_c2ll, boundbox_c1ur, boundbox_c2ur,  frm_name, frm_specificid, event_type, obs_channelid, name)
        
        #for NOAA SWPC Observer
        #save_event_information_noaa(event_starttime, event_endtime, hpc_bbox, hpc_boundcc, hpc_coord, hpc_radius, boundbox_c1ll, boundbox_c2ll, boundbox_c1ur, boundbox_c2ur,  frm_name, ar_noaanum, event_type, obs_channelid, name)
    
    
    else:
        event_starttime = res['event_starttime']
        #event_starttime = convert_est_to_utc(event_starttime)
        event_endtime = res['event_endtime']
        #event_endtime = convert_est_to_utc(event_endtime)
        hpc_bbox = res['hpc_bbox']
        hpc_boundcc = res['hpc_boundcc']
        hpc_coord = res['hpc_coord']
        boundbox_c1ll = res['boundbox_c1ll']
        boundbox_c2ll = res['boundbox_c2ll']
        boundbox_c1ur = res['boundbox_c1ur']
        boundbox_c2ur = res['boundbox_c2ur']
        frm_name = res['frm_name']
        frm_specificid = res['frm_specificid']
        #ar_noaanum = res['ar_noaanum']
        event_type = res['event_type']
        obs_channelid = res['obs_channelid']
        #name = 'SPoCA_22494'
        save_event_information(event_starttime, event_endtime, hpc_bbox, hpc_boundcc, hpc_coord, hpc_radius, boundbox_c1ll, boundbox_c2ll, boundbox_c1ur, boundbox_c2ur,  frm_name, frm_specificid, event_type, obs_channelid, excluded_name)
        
        
        #for NOAA SWPC Observer
        #save_event_information_noaa(event_starttime, event_endtime, hpc_bbox, hpc_boundcc, hpc_coord,  hpc_radius, boundbox_c1ll, boundbox_c2ll, boundbox_c1ur, boundbox_c2ur,  frm_name, ar_noaanum, event_type, obs_channelid, excluded_name)
        
       
df = pd.read_csv('csv/' + name + '.csv', index_col=0)

a = df.duplicated()
    


df = df.drop_duplicates()


#grouped = df.groupby('frm_specificid').filter(lambda x: len(x) >= 3)
grouped = df.groupby('frm_specificid')
gf = grouped.filter(lambda x: len(x['frm_name']) > 2.)
gf_grouped = gf.groupby('frm_specificid')

#gf_grouped['frm_specificid'].agg([np.sum])

grouped_data_save_directory = 'grouped_data/' + event_type+ '/'
if not os.path.isdir(grouped_data_save_directory):
    os.makedirs(grouped_data_save_directory)

for i, g in gf_grouped:
    #event_id = re.search(r'CH_\d{10}', i)
    #event_id = event_id.group(0)
    print (str(i))
    event_id = i.replace("SPoCA_v1.0_CH_", "")
    print('SPoCA' + '_' + event_id)
    g.to_csv('grouped_data/' + event_type + '/' + module + '_' + '{}.csv'.format(event_id), header=True)



'''
for res in results:
    print(res['hpc_boundcc'])
'''

