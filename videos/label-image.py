import numpy as np
import cv2


filename = '2.jpg'
# Create a black image
img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)


# Write some Text

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,200)
fontScale              = 5
fontColor              = (255,255,255)
lineType               = 2


iou = 0.46373834318796164

'''
cv2.putText(img,'Hello World!', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)
'''

cv2.putText(img, "IoU: {:.4f}".format(iou), (10, 200),
		cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)

#Display the image
#cv2.imshow("img",img)

#Save image
cv2.imwrite("out.jpg", img)

cv2.waitKey(0)