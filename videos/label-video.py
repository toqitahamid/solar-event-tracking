import numpy
import cv2

cap = cv2.VideoCapture('SPoCA_0000027536.avi')

i=2
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('SPoCA_0000027536_images/' + str(i)+'.jpg', frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()