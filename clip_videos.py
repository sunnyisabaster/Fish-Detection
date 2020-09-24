import cv2
import glob
import os
import time
import ntpath

video_folder = r'/data/project/content/drive/My Drive/testing/allResource/videos/'
frame_out_folder = r'/data/project/frame'

video_files = glob.glob(video_folder + "*.mp4")
video_files.sort()

ntpath.basename(video_files[0]).split('.')[0]

print(len(video_files), video_files[0])

# loop all video and create output folders

start_time = time.time()

for i, video_file in enumerate(video_files):

    start_time_v = time.time()

    # create outfolder
    out_folder = os.path.join(frame_out_folder, ntpath.basename(video_file).split('.')[0])
    try:

        # creating a folder named data
        # rise OSError
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # extract frame
        currentframe = 0

        cam = cv2.VideoCapture(video_file)

        while(True):

            # reading from frame
            ret,frame = cam.read()

            if ret:

                name = os.path.join(out_folder, str(currentframe) + '.jpg')
                # writing the extracted images
                cv2.imwrite(name, frame)
                currentframe += 1

            else:
                #read the end of the video
                break


        cam.release()

    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')


    print(str(i) + " --- %s seconds ---" % (time.time() - start_time_v))
    #break

print("total: %s seconds ---" % (time.time() - start_time))
