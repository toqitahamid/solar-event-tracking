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


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
def convert_jp2_to_other_format(jp2_img, save_location):
    img = Image(str(jp2_img))
    img.write(str(save_location))

parent_directory = sorted(glob.glob( 'jp2dump/AR/171/*.jp2'), key=numericalSort)


dest = 'jpgdump/AR/171/'
for jp2 in parent_directory:
    filename_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}__SDO_AIA_AIA_171', jp2)    
    filename = filename_match.group(0)
    jpg_filename = filename + ".jpg"
    save_location = dest + jpg_filename
    convert_jp2_to_other_format(jp2, save_location)

