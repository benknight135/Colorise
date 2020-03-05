import cv2 
import glob
import natsort
import os

def images_to_video(input_folder,output_file="vid.avi",fps=25,input_file_prefix="img"):
    video_file = input_folder+"/vid.avi"

    image_file_list = glob.glob(input_folder+"/"+input_file_prefix+'*.png')
    sorted_image_file_list = natsort.natsorted(image_file_list)

    img = cv2.imread(sorted_image_file_list[0])
    height = img.shape[0]
    width = img.shape[1]

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(video_file,fourcc, fps, (width,height))
    
    list_length = len(sorted_image_file_list)
    i = 0
    try:
        for image_file in sorted_image_file_list:
            complete = ((i+1)/list_length) * 100
            print("Reading frame {}... [{:.2f}%]".format(image_file,complete))
            img = cv2.imread(image_file)
            out.write(img)
            i += 1
    except KeyboardInterrupt:
        pass

    out.release()

def video_to_images(input_video,output_folder,output_file_prefix="img",start_frame=0,end_frame=None):

    if (end_frame is None):
        end_frame = 999999999
    # Read the video from specified path 
    cam = cv2.VideoCapture(input_video) 

    try: 
        
        # creating a folder named data 
        if not os.path.exists(output_folder): 
            os.makedirs(output_folder) 
    
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of input') 
    
    # frame 
    currentframe = 0
    totalframes = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    
    try:
        while(True): 
            
            # reading from frame 
            ret,frame = cam.read()

            complete = ((currentframe+1)/totalframes) * 100

            if (currentframe >= start_frame):
                if ret: 
                    # if video is still left continue creating images 
                    name = output_folder + "/" + output_file_prefix + str(currentframe) + '.png'
                    print ('Creating {}... {}/{} [{:.2f}%]'.format(name,currentframe,totalframes,complete)) 
            
                    # writing the extracted images 
                    cv2.imwrite(name, frame) 
            
                else: 
                    break
            else:
                print("skipping...{}/{} [{:.2f}%]".format(currentframe,start_frame,complete))
            
            currentframe += 1

            if currentframe > end_frame:
                break
    except KeyboardInterrupt:
        pass
    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 