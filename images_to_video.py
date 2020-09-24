import os
import cv2
import time

def picvideo(path,size):
    filelist = os.listdir(path)
    file_num = len(filelist) + 1
    #filelist.sort()
    #print(len(filelist))
    fps = 25

    #size = (1280,720)
    file_path = r'/data/project/'+str(int(time.time())) + ".mp4"
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')#'I','4','2','0' for avi

    video = cv2.VideoWriter( file_path, fourcc, fps, size )

    for item in range(0,file_num):
        #if item.endswith('.jpg'):
            item = path  + str(item) + '.jpg'
            #print(item)
            img = cv2.imread(item)
            video.write(img)

    video.release()

picvideo(r'/data/project/aja-helo-1H000314_2019-07-09_0904/',(1280,720))
