import numpy as np
import cv2

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


def annotation_area():
    boxAArea = (annPoly_x[3] - annPoly_x[1] ) * (annPoly_y[3] - annPoly_y[1] )

def bb_intersection_over_union(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
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
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (annPoly_x[3] - annPoly_x[1] + 1) * (annPoly_y[3] - annPoly_y[1] + 1)
	boxBArea = (trackedPoly_x[2] - trackedPoly_x[0] + 1) * (trackedPoly_y[2] - trackedPoly_y[0] + 1)
 
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
    
    overlap = max(xB - xA + 1, 0) * max(yB - yA + 1, 0)
    #print(overlap)
    closure = max(x_max - x_min + 1, 0) * max(y_max - y_min + 1, 0)
    #print(closure)
    #ann_area = polygon_area(annPoly_x, annPoly_y)
    ann_area = (annPoly_x[3] - annPoly_x[1] + 1) * (annPoly_y[3] - annPoly_y[1] + 1)
    #print(ann_area)
    #tracked_area = polyarea(trackedPoly_x, trackedPoly_y)
    tracked_area = (trackedPoly_x[2] - trackedPoly_x[0] + 1) * (trackedPoly_y[2] - trackedPoly_y[0] + 1)
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
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    #print(yB - yA)
    return interArea

def midpoint_ann(ptA, ptB):
	return ((ptA[3] + ptA[1]) * 0.5), ((ptB[3] + ptB[1]) * 0.5)

def midpoint_output(ptA, ptB):
	return ((ptA[2] + ptA[0]) * 0.5), ((ptB[2] + ptB[0]) * 0.5)


otp_errors = []  
def OTP_method(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y, union, sharedArea):
    if ((annPoly_x[1] == trackedPoly_x[0]) and (annPoly_y[1] == trackedPoly_y[0]) and (annPoly_x[3] == trackedPoly_x[2]) and (annPoly_y[3] == trackedPoly_y[2])):
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
    
    return l1_norm_distance_x+l1_norm_distance_y
    
def th(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
    width_ann = annPoly_x[3] - annPoly_x[1]
    height_ann = annPoly_y[3] - annPoly_y[1]
    
    width_output = trackedPoly_x[2] - trackedPoly_x[0]
    height_output = trackedPoly_y[2] - trackedPoly_y[0]
    
    th = (width_ann + height_ann + width_output + height_output)/2
    return th

pbm_list = []
def PBM(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y):
    
    
    if ((annPoly_x[1] == trackedPoly_x[0]) or (annPoly_y[1] == trackedPoly_y[0]) or (annPoly_x[3] == trackedPoly_x[2]) or (annPoly_y[3] == trackedPoly_y[2])):
        distance = l1_norm_distance(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        th_value = th(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        
        formula = 1-(distance/th_value)
        pbm_list.append(formula)
    else:
        distance = th(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        th_value = th(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
        formula = 1-(distance/th_value)
        pbm_list.append(formula)
    
def label_image(annFrameId, iou, iou_ann, iou_big):

    filename = 'SPoCA_0000027536_images/' + str(annFrameId) + '.jpg'
    # Create a black image
    img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
    
    
    # Write some Text
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,200)
    fontScale              = 5
    fontColor              = (255,255,255)
    #lineType               = 2
    
    
    '''
    cv2.putText(img,'Hello World!', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    '''
    
    cv2.putText(img, "IoU: {:.4f}".format(iou), (100, 200),
    		cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255), thickness=10, lineType=2)
    
    cv2.putText(img, "Covers Original Label: {:.4f}".format(iou_ann), (100, 400),
    		cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255), thickness=10, lineType=2)
    
    
    cv2.putText(img, "Predicted Label: {:.4f}x".format(iou_big), (100, 600),
    		cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255), thickness=10, lineType=2)
    
    #Display the image
    #cv2.imshow("img",img)
    
    #Save image
    cv2.imwrite('SPoCA_0000027536_labels/' + str(annFrameId)+'.jpg', img)
    
    cv2.waitKey(0)

outputFile = np.loadtxt("SPoCA_0000027536", dtype='float', delimiter=' ')
#print(outputFile)

outputFile[:, 4] = outputFile[:, 2] + outputFile[:, 4]
outputFile[:, 3] = outputFile[:, 1] + outputFile[:, 3]


ann_File = np.loadtxt("SPoCA_0000027536.ann", dtype='float', delimiter=' ')

NoAnnFrames = ann_File[:, 0].size
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
annotated_frames= 0
for i in range(NoAnnFrames):
    #print(i)
    
    ann = ann_File[i, :]
    #ann = trimDoubleZeros(ann)
    frameId = int(ann[0])
    annPoly_x = ann[1::2]
    annPoly_y = ann[2::2]
    
    # find the corresponding frame in the trackingResultFile
    corrFrameId = frameIndex[np.where(frameIndex == frameId)]
    
    if corrFrameId.size == 0:
        errors[i][0] = frameId
        errors[i][1] = float('nan')
        
        giou_errors[i][0] = frameId
        giou_errors[i][1] = float('nan')
        print('not found')
        continue
    #break;
    
    
    trackedPosIndex = np.where(frameIndex == frameId)
    trackedPosIndex = trackedPosIndex[0]
    
    trackedPos = outputFile[trackedPosIndex, 1:5]
    #trackedPos = outputFile[corrFrameId-2, 1:5]
    trackedPoly_x = np.zeros((1, 4))
    trackedPoly_x = [trackedPos[0][0], trackedPos[0][2], trackedPos[0][2], trackedPos[0][0]]
    trackedPoly_y = [trackedPos[0][1], trackedPos[0][1], trackedPos[0][3], trackedPos[0][3]]
    
    #compute the overlapping area
    
    #annArea = polygon_area(annPoly_x, annPoly_y);
    annArea = (annPoly_x[3] - annPoly_x[1] + 1) * (annPoly_y[3] - annPoly_y[1] + 1)
    
    
    #trackedArea = polyarea(trackedPoly_x, trackedPoly_y)
    trackedArea = (trackedPoly_x[2] - trackedPoly_x[0] + 1) * (trackedPoly_y[2] - trackedPoly_y[0] + 1)
    
    sharedArea = bb_intersection_area(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
    
    union = bb_union(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
    iou = bb_intersection_over_union(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
    iou_ann = bb_intersection_over_union_ann(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
    
    iou_big = bb_intersection_over_union_big(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
    
    label_image(corrFrameId[0], iou, iou_ann, iou_big)
    
    giou_value = giou_method(annPoly_x, annPoly_y, trackedPoly_x, trackedPoly_y)
    #print(iou)
  
    
    
    errors[i][0] = frameId
    
    #errors[i][1] = sharedArea / (annArea + trackedArea - sharedArea)
    errors[i][1] = iou
    
    

    
    annotated_frames += 1
    
    #print(errors)
    #print('----------------------------------------')
    
    
#FSCORE 
theta = 0.5
#overlaps = errors[:, 1]

overlaps = errors[:, 1]
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

if P != 0.0 or R != 0.0:
    fscore = (2*P*R)/(P+R);
else:
    fscore = 0.0
#fscores += fscore
    
#fscores = fscores/NoAnnFrames
print('Fscore: ' + str(fscore))


