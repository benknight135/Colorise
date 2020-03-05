import ml_colorise
import image_convert
import os

# VIDEO ML
# create output folder if it does not exist
if not os.path.exists("output"): 
    os.makedirs("output")
ml_colorise.test_video("input/vid.mp4","output/vid.mp4","tmp",ml_colorise.ML_TYPE_TF,ml_colorise.ML_MODE_COLORIZE)
ml_colorise.test_video("output/vid.mp4","output_super/vid.mp4","tmp",ml_colorise.ML_TYPE_DEEPAI,ml_colorise.ML_MODE_SUPER_RES)