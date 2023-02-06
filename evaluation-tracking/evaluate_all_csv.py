import numpy as np
import glob
import re



numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def trimDoubleZeros(ann):
    i = ann.size
    while ann[i-1] == 0 and i > 0:
        if abs(ann[i-1] + ann[i-2]) == 0:
             ann = np.delete(ann, i-1)
             ann = np.delete(ann, i-2)
             i = i - 1;
        i = i - 1;


def polygon_area(x,y):
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)

def polyarea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def bb_intersection_over_union(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(annPoly_x[1], trackedPoly_x[0])
	yA = max(annPoly_y[1], trackedPoly_y[0])
	xB = min(annPoly_x[3], trackedPoly_x[2])
	yB = min(annPoly_y[3], trackedPoly_y[2])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA  + 1) * max(0, yB - yA  + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (annPoly_x[3] - annPoly_x[1]  + 1) * (annPoly_y[3] - annPoly_y[1]  + 1)
	boxBArea = (trackedPoly_x[2] - trackedPoly_x[0]  + 1) * (trackedPoly_y[2] - trackedPoly_y[0]  + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


def bb_intersection_over_union_ann(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(annPoly_x[1], trackedPoly_x[0])
	yA = max(annPoly_y[1], trackedPoly_y[0])
	xB = min(annPoly_x[3], trackedPoly_x[2])
	yB = min(annPoly_y[3], trackedPoly_y[2])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA  + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (annPoly_x[3] - annPoly_x[1] + 1) * (annPoly_y[3] - annPoly_y[1] + 1)
	boxBArea = (trackedPoly_x[2] - trackedPoly_x[0] + 1) * (trackedPoly_y[2] - trackedPoly_y[0] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou_ann = interArea / float(boxAArea)
 
	# return the intersection over union value
	return iou_ann


def bb_intersection_over_union_big(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(annPoly_x[1], trackedPoly_x[0])
	yA = max(annPoly_y[1], trackedPoly_y[0])
	xB = min(annPoly_x[3], trackedPoly_x[2])
	yB = min(annPoly_y[3], trackedPoly_y[2])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA  + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (annPoly_x[3] - annPoly_x[1] + 1) * (annPoly_y[3] - annPoly_y[1] + 1)
	boxBArea = (trackedPoly_x[2] - trackedPoly_x[0] + 1) * (trackedPoly_y[2] - trackedPoly_y[0] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou_big = float(boxBArea) / float(boxAArea)
 
	# return the intersection over union value
	return iou_big


def bb_union(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(annPoly_x[1], trackedPoly_x[0])
	yA = max(annPoly_y[1], trackedPoly_y[0])
	xB = min(annPoly_x[3], trackedPoly_x[2])
	yB = min(annPoly_y[3], trackedPoly_y[2])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA  + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (annPoly_x[3] - annPoly_x[1]  + 1) * (annPoly_y[3] - annPoly_y[1]  + 1)
	boxBArea = (trackedPoly_x[2] - trackedPoly_x[0]  + 1) * (trackedPoly_y[2] - trackedPoly_y[0]  + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	union =  float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return union


def giou_method(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(annPoly_x[1], trackedPoly_x[0])
    yA = max(annPoly_y[1], trackedPoly_y[0])
    xB = min(annPoly_x[3], trackedPoly_x[2])
    yB = min(annPoly_y[3], trackedPoly_y[2])
    
    x_min = min(annPoly_x[1], trackedPoly_x[0])
    y_min = min(annPoly_y[1], trackedPoly_y[0])
    
    x_max = max(annPoly_x[3], trackedPoly_x[2])
    y_max = max(annPoly_y[3], trackedPoly_y[2])
    
    # compute the area of intersection rectangle
#    interArea = max(0, xB - xA) * max(0, yB - yA)
    #print(yB - yA)
    
    #overlap = np.maximum(xB - xA + 1, 0) * np.maximum(yB - yA + 1, 0)
    #closure = np.maximum(x_max - x_min + 1, 0) * np.maximum(y_max - y_min + 1, 0)
    
    overlap = max(xB - xA , 0) * max(yB - yA , 0)
    #print(overlap)
    closure = max(x_max - x_min , 0) * max(y_max - y_min , 0)
    #print(closure)
    #ann_area = polygon_area(annPoly_x, annPoly_y)
    ann_area = (annPoly_x[3] - annPoly_x[1] ) * (annPoly_y[3] - annPoly_y[1])
    #print(ann_area)
    #tracked_area = polyarea(trackedPoly_x, trackedPoly_y)
    tracked_area = (trackedPoly_x[2] - trackedPoly_x[0] ) * (trackedPoly_y[2] - trackedPoly_y[0] )
    #print(tracked_area)
    union = ann_area + tracked_area - overlap
    closure
    gious = overlap / union - (closure - union) / closure
    
    
    return gious


def bb_intersection_area(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(annPoly_x[1], trackedPoly_x[0])
    yA = max(annPoly_y[1], trackedPoly_y[0])
    xB = min(annPoly_x[3], trackedPoly_x[2])
    yB = min(annPoly_y[3], trackedPoly_y[2])
    #print(yA)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    #print(yB - yA)
    return interArea

def midpoint_ann(ptA, ptB):
	return ((ptA[3] + ptA[1]) * 0.5), ((ptB[3] + ptB[1]) * 0.5)

def midpoint_output(ptA, ptB):
	return ((ptA[2] + ptA[0]) * 0.5), ((ptB[2] + ptB[0]) * 0.5)


otp_errors = [] 
def OTP_method(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y, union, sharedArea, iou):
    #if ((annPoly_x[1] == trackedPoly_x[0]) and (annPoly_y[1] == trackedPoly_y[0]) and (annPoly_x[3] == trackedPoly_x[2]) and (annPoly_y[3] == trackedPoly_y[2])):
    if (iou >= 0.99 ):
        
        otp_errors.append(abs(sharedArea)/abs(union))

centroid_normalized_distance = []

def deviation_method(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
    if ((annPoly_x[1] == trackedPoly_x[0]) and (annPoly_y[1] == trackedPoly_y[0]) and (annPoly_x[3] == trackedPoly_x[2]) and (annPoly_y[3] == trackedPoly_y[2])):
            
        midpoint_ann_x, midpoint_ann_y = midpoint_ann(annPoly_x, annPoly_y)
        #midpoint_ann[i][1] = midpoint_ann_x
        #midpoint_ann[i][2] = midpoint_ann_y
        
        midpoint_output_x, midpoint_output_y = midpoint_output(trackedPoly_x, trackedPoly_y)
        #midpoint_output[i][1] = midpoint_output_x
        #midpoint_output[i][2] = midpoint_output_y
        centroid_normalized_distance_x = np.square(midpoint_output_x - midpoint_ann_x)
        centroid_normalized_distance_y = np.square(midpoint_output_y - midpoint_ann_y)
        centroid_normalized_distance.append(np.sqrt(centroid_normalized_distance_x + centroid_normalized_distance_y))
    
    
def l1_norm_distance(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
    midpoint_ann_x, midpoint_ann_y = midpoint_ann(annPoly_x, annPoly_y)
    midpoint_output_x, midpoint_output_y = midpoint_output(trackedPoly_x, trackedPoly_y)
    l1_norm_distance_x = midpoint_ann_x -  midpoint_output_x
    l1_norm_distance_y = midpoint_ann_y -  midpoint_output_y
    
    #l1_norm_distance_x = midpoint_output_x - midpoint_ann_x
    #l1_norm_distance_y = midpoint_output_y - midpoint_ann_y
    
    #print("l1_norm_distance_x" + str(l1_norm_distance_x))
    
    return l1_norm_distance_x+l1_norm_distance_y
    
def th(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
    width_ann = annPoly_x[3] - annPoly_x[1]
    height_ann = annPoly_y[3] - annPoly_y[1]
    
    width_output = trackedPoly_x[2] - trackedPoly_x[0]
    height_output = trackedPoly_y[2] - trackedPoly_y[0]
    
    th = (width_ann + height_ann + width_output + height_output)/2
    return th

pbm_list = []
def PBM(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y, sharedArea):
    
    if (sharedArea > 0.0):
    #if ((annPoly_x[1] == trackedPoly_x[0]) or (annPoly_y[1] == trackedPoly_y[0]) or (annPoly_x[3] == trackedPoly_x[2]) or (annPoly_y[3] == trackedPoly_y[2])):
        distance = l1_norm_distance(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        th_value = th(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        
        formula = 1-(distance/th_value)
        if formula > 2:
            formula = 0
        pbm_list.append(formula)
    else:
        distance = th(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        th_value = th(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        formula = 1-(distance/th_value)
        if formula > 2:
            formula = 0
        pbm_list.append(formula)
    
errors = []

error_list = []
TP_all = 0
ann_total = 0
iou_all = []
giou_value_list = []
def evaluate(output_file, annotation_file, theta):
    outputFile = np.loadtxt(output_file, dtype='float', delimiter=' ')
    #a.append(outputFile)
    #print(outputFile)
    
    outputFile[:, 4] = outputFile[:, 2] + outputFile[:, 4]
    outputFile[:, 3] = outputFile[:, 1] + outputFile[:, 3]
    
    
    ann_File = np.loadtxt(annotation_file, dtype='float', delimiter=' ')
    #print(ann_File.shape)
    NoAnnFrames = ann_File[:, 0].size
    #print(NoAnnFrames)
    frameIndex = outputFile[: ,0]
    frameIndex = frameIndex.astype(int)
    w, h = 2, NoAnnFrames;
    errors = [ [0 for x in range( w )] for y in range( h ) ] 
        
    errors = np.array(errors ,dtype="float")
    
    giou_errors = [ [0 for x in range( w )] for y in range( h ) ] 
    
    giou_errors = np.array(giou_errors ,dtype="float")  
    
    
    w_1, h_1 = 3, NoAnnFrames
    area_based_errors = [ [0 for x in range( w_1 )] for y in range( h_1 ) ] 
    area_based_errors = np.array(area_based_errors ,dtype="float")
    
    
    w_2, h_2 = 3, NoAnnFrames
    ata_errors = [ [0 for x in range( w_2 )] for y in range( h_2 ) ] 
    ata_errors = np.array(ata_errors ,dtype="float")
    
    midpoint = [ [0 for x in range( w_2 )] for y in range( h_2 ) ] 
    midpoint = np.array(midpoint,dtype="float")
    
    
    
        
    

    #print(errors)
    #print('----------------------------------------')
    fscores = 0
    annotated_frames = 0
    for i in range(NoAnnFrames):
        #print( range(NoAnnFrames))
        
        ann = ann_File[i, :]
        #print(ann)
        #ann = trimDoubleZeros(ann)
        frameId = int(ann[0])
        annPoly_x = ann[1::2]
        annPoly_y = ann[2::2]
        #print(frameId)
        
        # find the corresponding frame in the trackingResultFile
        corrFrameId = frameIndex[np.where(frameIndex == frameId)]
        #print(corrFrameId)
        if corrFrameId.size == 0:
            errors[i][0] = frameId
            errors[i][1] = float('nan')
            
            giou_errors[i][0] = frameId
            giou_errors[i][1] = float('nan')
            
            
            area_based_errors[i][0] = frameId
            area_based_errors[i][1] = float('nan')
            area_based_errors[i][2] = float('nan')
            
            ata_errors[i][0] = frameId
            ata_errors[i][1] = float('nan')
            ata_errors[i][2] = float('nan')
            
            midpoint[i][0] = frameId
            midpoint[i][1] = float('nan')
            midpoint[i][2] = float('nan')
            
            #print('not found')
            continue
        
        trackedPosIndex = np.where(frameIndex == frameId)
        trackedPosIndex = trackedPosIndex[0]
        
        trackedPos = outputFile[trackedPosIndex, 1:5]
        #print(trackedPos)
        #trackedPos = outputFile[corrFrameId-1, 1:5]
        trackedPoly_x = np.zeros((1, 4))
        trackedPoly_x = [trackedPos[0][0], trackedPos[0][2], trackedPos[0][2], trackedPos[0][0]]
        trackedPoly_y = [trackedPos[0][1], trackedPos[0][1], trackedPos[0][3], trackedPos[0][3]]
        
        #compute the overlapping area
        
        #annArea = polygon_area(annPoly_x, annPoly_y)
        annArea = (annPoly_x[3] - annPoly_x[1] + 1) * (annPoly_y[3] - annPoly_y[1] + 1)
        #print(annArea)
        #trackedArea = polyarea(trackedPoly_x, trackedPoly_y)
        trackedArea = (trackedPoly_x[2] - trackedPoly_x[0] + 1) * (trackedPoly_y[2] - trackedPoly_y[0] + 1)
        
        #print(trackedArea)
        sharedArea = bb_intersection_area(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        #print(annPoly_x)
        union = bb_union(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        #iou = bb_intersection_over_union(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        iou = bb_intersection_over_union_ann(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        iou_all.append(iou)
        #print(iou)
        
        giou_value = giou_method(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        giou_value_list.append(giou_value)
        
        
        errors[i][0] = frameId
        
        #errors[i][1] = sharedArea / (annArea + trackedArea - sharedArea)
        errors[i][1] = sharedArea / (annArea)
        #errors[i][1] = iou
        #print(errors[i][1])
        error_list.append(sharedArea / (annArea))
        
        giou_errors[i][0] = frameId
    
        giou_errors[i][1] = giou_value
        
        area_based_errors[i][0] = frameId
        area_based_errors[i][1] = sharedArea / trackedArea #p_i
        area_based_errors[i][2] = sharedArea / annArea #r_i
        
        ata_errors[i][0] = frameId
        ata_errors[i][1] = sharedArea
        ata_errors[i][2] = union
        
        midpoint[i][0] = frameId
        midpoint_x, midpoint_y = midpoint_ann(annPoly_x, annPoly_y)
        midpoint[i][1] = midpoint_x
        midpoint[i][2] = midpoint_y
        
        
        
        OTP_method(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y, union, sharedArea, iou)    
        deviation_method(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        PBM(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y, sharedArea)
        
        annotated_frames += 1
        
        #print(errors)
        #print('----------------------------------------')
        
    
    #FSCORE 
    #theta = 0.5
    #e1 = errors
    overlaps = errors[:, 1]
    
    #overlaps = giou_errors[:, 1]
    
    #print("overlaps")
    #print(overlaps)
    #errors = sharedArea / (annArea + trackedArea - sharedArea)
    #overlaps = errors
    FN = np.count_nonzero(np.isnan(overlaps)) #no track box is ass. with a gt
    if np.isnan(np.sum(overlaps)):
        overlaps = overlaps[~np.isnan(overlaps)]
    TP = np.count_nonzero(overlaps >= theta)
    FP = np.count_nonzero(overlaps < theta) # a track box is not ass. with a gt
    FN = FN + FP # cvpr12 formula
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    
    global TP_all, ann_total
    TP_all +=  TP
    ann_total += NoAnnFrames
    if P != 0.0 or R != 0.0:
        fscore = (2*P*R)/(P+R);
    else:
        fscore = 0.0
    #fscores += fscore
        
    #fscores = fscores/NoAnnFrames
    #print('Fscore: ' + str(fscore))
    
    
    p_i = area_based_errors[:, 1]
    r_i = area_based_errors[:, 2]
    
    #if np.isnan(np.sum(p_i)):
    #    p_i = p_i[~np.isnan(p_i)]
        
    #if np.isnan(np.sum(r_i)):
    #    r_i = r_i[~np.isnan(r_i)]
    
    
    #AREA BASED F1
    p_r_mul = p_i * r_i
    p_r_sum = p_i + r_i
    p_r_2 = 2 * (p_r_mul/p_r_sum)
    
    #fix for RuntimeWarning: invalid value encountered in true_divide
    if np.isnan(np.sum(p_r_2)):
        p_r_2 = p_r_2[~np.isnan(p_r_2)]
    
    
    p_r_2_sum = np.sum(p_r_2)
    area_based_f1 = (1/NoAnnFrames) * p_r_2_sum
    #area_based_f1 = (1/annotated_frames) * p_r_2_sum
    
    #print('Area Based F-score: ' + str(area_based_f1))
    
    #OTA
    #print("FN+FP: " + str(FN+FP))
    #print("NoAnnFrames" + str(NoAnnFrames))
    #ota = 1 - ((FN+FP)/NoAnnFrames)
    ota = 1 - ((FN)/NoAnnFrames)
    #ota = 1 - ((FN)/annotated_frames)
    #print('OTA: '+ str(ota))
    
    Ms = 0
    
    #OTP
    otp_value = 0
    if (len(otp_errors) >= 1):
        sum_otp_errors = np.sum(otp_errors)
        Ms = len(otp_errors)
        OTP = (1/abs(Ms)) * sum_otp_errors
        #print ("OTP: " + str(OTP))
        otp_value = OTP
    
    #ATA
    intersection = ata_errors[:, 1]
    union = ata_errors[:, 2]
    intersection_by_union = abs(intersection)/abs(union )
    
    if np.isnan(np.sum(intersection_by_union)):
        intersection_by_union = intersection_by_union[~np.isnan(intersection_by_union)]
    
    
    intersection_by_union_sum = np.sum(intersection_by_union)
    ata = (1/NoAnnFrames) * intersection_by_union_sum
    #print('ATA: ' + str(ata))
    
    #Deviation
    deviation_value = 0
    if len(centroid_normalized_distance) >= 1:
            
        deviation = 1 - (sum(centroid_normalized_distance)/abs(Ms))
        #print('Deviation: ' + str(deviation))
        deviation_value = deviation
    
    #PBM
    pbm = (1/NoAnnFrames) * sum(pbm_list)
    #print("PBM: " + str(pbm))

    return fscore, area_based_f1, ota, otp_value, ata, deviation_value, pbm


output_folder = sorted(glob.glob('ch_14_17/*'), key=numericalSort)
annotations_folder = "solar_annotation"
fscore_list = []
area_based_f1_list = []
ota_list = []
OTP_list = [] 
ata_list = []
deviation_list = []
pbm_list  = []


categories = []
threshes = [0.5, 0.7, 0.9]
theta = 0.5
scores = np.zeros(shape=(len(threshes) ,1), dtype= float)


for i in output_folder:
    #print(i)
    output_file = i
    
    filename = ''
    category = ''
    if re.search(r'HMI_\d{4}', output_file) is not None:
        if re.search(r'HMI_\d{4}', output_file).group(0).split("_")[0] == 'HMI':
            category = "AR"
            filename = re.search(r'HMI_\d{4}', output_file).group(0)
    elif re.search(r'SPoCA_\d{10}', output_file) is not None:
        if re.search(r'SPoCA_\d{10}', output_file).group(0).split("_")[0] == 'SPoCA':
            category = "CH"
            filename = re.search(r'SPoCA_\d{10}', output_file).group(0)
    '''
    if re.search(r'HMI_\d{4}', output_file).group(0).split("_")[0] == 'HMI':
        category = "AR"
        filename = re.search(r'HMI_\d{4}', output_file).group(0)
    elif re.search(r'SPoCA_\d{10}', output_file).group(0).split("_")[0] == 'SPoCA':
        category = "CH"
        filename = re.search(r'SPoCA_\d{10}', output_file).group(0)
        
   
    first_match = re.search(r'HMI_\d{4}', output_file)
    filename = first_match.group(0)
    split_filename = filename.split("_")
    category = split_filename[0]
    print(filename)
    
    if category == "HMI":
        category = "AR"
    elif category == "SPoCA":
        category = "AR"
    '''
    
    
    #output_file = output_folder[1]
     
    annotation_file = annotations_folder + '/' + category + '/' + filename + '.ann'
    
    fscore_val, area_based_f1_val, ota_val, OTP_val, ata_val, deviation_val, pbm_val = evaluate(output_file, annotation_file, theta)
    
    fscore_list.append(fscore_val)
    area_based_f1_list.append(area_based_f1_val)
    ota_list.append(ota_val)
    OTP_list.append(OTP_val) 
    ata_list.append(ata_val)
    deviation_list.append(deviation_val)
    pbm_list.append(pbm_val)
    
print("Fscore Mean: " + str(np.mean(fscore_list)))
print("Area Based F1 Mean: " + str(np.mean(area_based_f1_list)))
print("OTA Mean: " + str(np.mean(ota_list)))
print("OTP Mean: " + str(np.mean(OTP_val)))
print("ATA Mean: " + str(np.mean(ata_val)))
print("Deviation Mean: " + str(np.mean(deviation_val)))
print("PBM Mean: " + str(np.mean(pbm_val)))

    #print(Fscore)

'''
Fscores = [];
categories = [];

threshes = [0.5 0.7 0.9];
scores = zeros(length(threshes),1);

for j = 1:length(threshes)
    thresh = threshes(j);
    for i = 1:length(output_files)
        file = output_files(i);
        
        if (file.isdir)
            continue;
        end
        
        if (file.name(1) == '.')
            continue
        end
        


        categories = [categories category];
        

        output_file = [output_folder '/' file.name];
        
        annotation_file = [annotations_folder '/' category '/' file.name '.ann'];     
        
        Fscore = quantitativeEvaluationFScore_poly(output_file, annotation_file, thresh);
        
        if (isnan(Fscore))
            Fscore = 0;
        end      
        Fscores = [Fscores; Fscore];
    end
    scores(j) = mean(Fscores);
end

for i = 1:length(threshes)
   fprintf('Thresh: %f, Mean: %f\n', threshes(i), scores(i)); 
end

end


'''


