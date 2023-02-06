import csv
import pandas as pd
import datetime

from sunpy.net.helioviewer import HelioviewerClient
import sunpy.io.jp2 as jp2
hv = HelioviewerClient()


csv_file = 'image_file_not_found.csv'

data = pd.read_csv(csv_file, usecols=['filename'])
directoryJp2 = 'image_not_found/'        
        
filename = data.filename.tolist()

def create_events_with_filename_header(save_dir, event_name):
    #data = [[image_counter, bbox, boundcc,  cdelt1, cdelt2, crpix1, crpix2, filename]]
    event_filename = str(event_name)+'.csv'
    
    save_path = os.path.join(save_dir, event_filename)
    
    with open(save_path, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['image_counter', 'filename'])
        #writer.writerows(data)
    
    csvFile.close()   

for i in filename:
    

    start = datetime.datetime.strptime(i, "%Y_%m_%d__%H_%M_%S_%f__SDO_AIA_AIA_171.jp2")

    time = start.replace(microsecond=0)
    
    print(time)
    
    jp2_location = hv.download_jp2(start, observatory='SDO', instrument='AIA', detector='SPoCA', measurement='171', sourceId = '10', directory=directoryJp2)