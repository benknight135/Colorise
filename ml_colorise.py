# deepai dependencies
import requests
import os
import glob
import natsort
# tensorflow ml dependencies
import keras.backend as K
import numpy as np
import sklearn.neighbors as nn
import cv2
# custom tensorflow helper scripts
from config import img_rows, img_cols
from config import nb_neighbors, T, epsilon
from model import build_model

# global variables
DEEPAI_MODEL_COLORIZE = 'colorizer'
DEEPAI_MODEL_SUPER_RES = 'torch-srgan'
ML_TYPE_TF = "ML_TYPE_TF"
ML_TYPE_DEEPAI = "ML_TYPE_DEEPAI"
ML_MODE_COLORIZE = "ML_MODE_COLORIZE"
ML_MODE_SUPER_RES = "ML_MODE_SUPER_RES"

class TF_COLORISE():
    def __init__(self):
        pass

    def pre_load_model(self,ml_mode=ML_MODE_COLORIZE):
        if (ml_mode == ML_MODE_COLORIZE):
            model_weights_path='models/model.06-2.5489.hdf5'
            self.model = build_model()
            self.model.load_weights(model_weights_path)

            print(self.model.summary())

            # Load the array of quantized ab value
            self.q_ab = np.load("data/pts_in_hull.npy")
            self.nb_q = self.q_ab.shape[0]

            # Fit a NN to q_ab
            self.nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(self.q_ab)
        elif (ml_mode == ML_MODE_SUPER_RES):
            print("Super res not yet implimented in TF_COLORISE class")
        else:
            print("Invalid ML Mode: ",ml_mode)

    def run(self,in_img_file,out_img_file):
        h, w = img_rows // 4, img_cols // 4

        filename = in_img_file
        #print('Start processing image: {}'.format(filename))
        # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
        bgr = cv2.imread(filename)

        i_img_rows, i_img_cols = bgr.shape[:2]

        gray = cv2.imread(filename, 0)
        bgr = cv2.resize(bgr, (img_cols, img_rows), cv2.INTER_CUBIC)
        gray = cv2.resize(gray, (img_cols, img_rows), cv2.INTER_CUBIC)
        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]
        # print('np.max(L): ' + str(np.max(L)))
        # print('np.min(L): ' + str(np.min(L)))
        # print('np.max(a): ' + str(np.max(a)))
        # print('np.min(a): ' + str(np.min(a)))
        # print('np.max(b): ' + str(np.max(b)))
        # print('np.min(b): ' + str(np.min(b)))
        x_test = np.empty((1, img_rows, img_cols, 1), dtype=np.float32)
        x_test[0, :, :, 0] = gray / 255.

        # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
        X_colorized = self.model.predict(x_test)
        X_colorized = X_colorized.reshape((h * w, self.nb_q))

        # Reweight probas
        X_colorized = np.exp(np.log(X_colorized + epsilon) / T)
        X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

        # Reweighted
        q_a = self.q_ab[:, 0].reshape((1, 313))
        q_b = self.q_ab[:, 1].reshape((1, 313))

        X_a = np.sum(X_colorized * q_a, 1).reshape((h, w))
        X_b = np.sum(X_colorized * q_b, 1).reshape((h, w))
        # print('np.max(X_a): ' + str(np.max(X_a)))
        # print('np.min(X_a): ' + str(np.min(X_a)))
        # print('np.max(X_b): ' + str(np.max(X_b)))
        # print('np.min(X_b): ' + str(np.min(X_b)))
        X_a = cv2.resize(X_a, (img_cols, img_rows), cv2.INTER_CUBIC)
        X_b = cv2.resize(X_b, (img_cols, img_rows), cv2.INTER_CUBIC)

        # Before: -90 <=a<= 100, -110 <=b<= 110
        # After: 38 <=a<= 228, 18 <=b<= 238
        X_a = X_a + 128
        X_b = X_b + 128
        # print('np.max(X_a): ' + str(np.max(X_a)))
        # print('np.min(X_a): ' + str(np.min(X_a)))
        # print('np.max(X_b): ' + str(np.max(X_b)))
        # print('np.min(X_b): ' + str(np.min(X_b)))

        out_lab = np.zeros((img_rows, img_cols, 3), dtype=np.int32)
        out_lab[:, :, 0] = lab[:, :, 0]
        out_lab[:, :, 1] = X_a
        out_lab[:, :, 2] = X_b
        out_L = out_lab[:, :, 0]
        out_a = out_lab[:, :, 1]
        out_b = out_lab[:, :, 2]
        # print('np.max(out_L): ' + str(np.max(out_L)))
        # print('np.min(out_L): ' + str(np.min(out_L)))
        # print('np.max(out_a): ' + str(np.max(out_a)))
        # print('np.min(out_a): ' + str(np.min(out_a)))
        # print('np.max(out_b): ' + str(np.max(out_b)))
        # print('np.min(out_b): ' + str(np.min(out_b)))
        out_lab = out_lab.astype(np.uint8)
        out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
        # print('np.max(out_bgr): ' + str(np.max(out_bgr)))
        # print('np.min(out_bgr): ' + str(np.min(out_bgr)))
        out_bgr = out_bgr.astype(np.uint8)

        out_bgr_resize = cv2.resize(out_bgr, (i_img_cols,i_img_rows), cv2.INTER_CUBIC)

        cv2.imwrite(out_img_file, out_bgr_resize)
        return out_bgr_resize

tf_col = TF_COLORISE()

def deepai(in_img_file,out_img_file,ml_mode=ML_MODE_COLORIZE):
    model = None
    if (ml_mode == ML_MODE_COLORIZE):
        model = DEEPAI_MODEL_COLORIZE
    elif (ml_mode == ML_MODE_SUPER_RES):
        model = DEEPAI_MODEL_SUPER_RES
    else:
        print("Invalid ML Mode: ", ml_mode)

    r = requests.post(
        "https://api.deepai.org/api/"+model,
        files={
            'image': open(in_img_file, 'rb'),
        },
        headers={'api-key': 'a8c84881-c8f1-48d2-aed6-c585f43d6685'}
    )
    pic_url = r.json()['output_url']

    with open(out_img_file,'wb') as handle:
        response = requests.get(pic_url, stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)

    img = cv2.imread(out_img_file)
    return img

def tf(in_img_file,out_img_file,ml_mode=ML_MODE_COLORIZE):
    tf_col.run(in_img_file,out_img_file)

def test_video(input_file,output_file,tmp_folder="tmp",ml_type=ML_TYPE_DEEPAI,ml_mode=ML_MODE_COLORIZE,start_frame=0,end_frame=None):
    if (ml_type == ML_TYPE_TF):
        tf_col.pre_load_model(ml_mode)
    
    try:
        # creating a folder named data 
        if not os.path.exists(tmp_folder): 
            os.makedirs(tmp_folder)
    
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directorys')

    print("Processing images with " + ml_type + " for : " + ml_mode)

    if (end_frame is None):
        end_frame = 999999999
    # Read the video from specified path 
    cam = cv2.VideoCapture(input_file)

    if cam.isOpened(): 
        # get vcap property 
        width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        fps = cam.get(cv2.CAP_PROP_FPS)
        totalframes = int(cam.get(cv2.CAP_PROP_FRAME_COUNT)) 

        print(width,height)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(output_file,fourcc, fps, (width,height))
        
        # frame 
        currentframe = 0
        
        while(True): 
            
            # reading from frame 
            ret,frame = cam.read()

            complete = ((currentframe+1)/totalframes) * 100

            if (currentframe >= start_frame):
                if ret: 
                    # if video is still left continue creating images 
                    name = tmp_folder + "/img.png"
                    print ('Creating {}... {}/{} [{:.2f}%]'.format(name,currentframe,totalframes,complete)) 
            
                    # writing the extracted images 
                    cv2.imwrite(name, frame)

                    img_ml = None
                    if (ml_type == ML_TYPE_DEEPAI):
                        img_ml = deepai(name,tmp_folder+'/img_col.png',ml_mode)
                    elif (ml_type == ML_TYPE_TF):
                        img_ml = tf_col.run(name,tmp_folder+'/img_col.png')
                    else:
                        print("Invalid ML Type: ",ml_type)
                        break

                    out.write(img_ml)
            
                else: 
                    break
            else:
                print("skipping...{}/{} [{:.2f}%]".format(currentframe,start_frame,complete))
            
            currentframe += 1

            if currentframe > end_frame:
                break
        
        # Release all space and windows once done 
        cam.release()
        out.release()
        cv2.destroyAllWindows() 

def test(input_folder,output_folder,ml_type=ML_TYPE_DEEPAI,ml_mode=ML_MODE_COLORIZE,skip_files=0,input_file_prefix="img"):
    if (ml_type == ML_TYPE_TF):
        tf_col.pre_load_model(ml_mode)
    
    try:
        # creating a folder named data 
        if not os.path.exists(output_folder): 
            os.makedirs(output_folder)
    
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directorys')

    print("Processing images with " + ml_type + " for : " + ml_mode)

    image_file_list = glob.glob(input_folder+"/"+input_file_prefix+'*.png')
    sorted_image_file_list = natsort.natsorted(image_file_list)

    i = 0
    list_length = len(sorted_image_file_list)
    for img_file_path in sorted_image_file_list:
        complete = ((i+1)/list_length) * 100
        if i >= skip_files:
            img_filename = os.path.basename(img_file_path)
            print("Processing {}... [{:.2f}%]".format(img_filename,complete))
            if (ml_type == ML_TYPE_DEEPAI):
                deepai(img_file_path,output_folder+'/'+img_filename,ml_mode)
            elif (ml_type == ML_TYPE_TF):
                tf_col.run(img_file_path,output_folder+'/'+img_filename)
            else:
                print("Invalid ML Type: ",ml_type)
                break
        i += 1