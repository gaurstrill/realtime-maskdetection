from cv2 import LINE_AA
import cv2 as cv

cam=cv.VideoCapture(0)

while(True):
    #storing each frame in var frames
    ret,frames= cam.read()
    #grayscale
    gray= cv.cvtColor(frames,cv.COLOR_BGR2GRAY)

    '''#cv.imshow('webcam',frames)
    #cv.imshow('gray',gray)'''

    #importing classifier
    haarface=cv.CascadeClassifier('haarfrontface.xml')
    haarmouth=cv.CascadeClassifier('haarmouth.xml')

    #stroring co ordinates
    face_cord=haarface.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    mouth_cord=haarmouth.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=80)

    #mask detection and placing text
    if len(mouth_cord)>0:
        for(h,j,k,l) in mouth_cord:
            '''cv.rectangle(frames,(h,j),(h+k,j+l),(0,0,250), thickness=1)'''
            cv.putText(frames,'NOT WEARNING MASK',org=(h+k,j+l),fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0,0,255),thickness=1,lineType=LINE_AA)
    else:
        for(a,b,c,d) in face_cord:
            cv.putText(frames,'WEARNING MASK',org=(a+c
            ,b+d),fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(255,60,0),thickness=1,lineType=LINE_AA)

    cv.putText(frames,'MASK DETECTOR',org=(400,30),fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(0,0,0),thickness=1,lineType=LINE_AA)
    cv.putText(frames,'Look directly into the camera for better result.',org=(30,30),fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.4, color=(0,255,255),thickness=1,lineType=LINE_AA)

    
    '''for(a,b,c,d) in face_cord:
       # cv.rectangle(frames,(a,b),(a+c,b+d),(0,250,250), thickness=1)'''
    
    
    cv.imshow('DETECTION',frames)

    #pressing esc to exit 
    c=cv.waitKey(1)
    if c==27:
        break
    
cam.release()
cv.destroyAllWindows()
