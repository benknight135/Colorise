import ml_colorise
import image_convert
import os

# IMAGE ML
#image_convert.video_to_images("input/vid.mp4","input")
#ml_colorise.test("input","output",ml_colorise.ML_TYPE_TF,ml_colorise.ML_MODE_COLORIZE)
#image_convert.images_to_video("output")

# VIDEO ML
# create output folder if it does not exist
if not os.path.exists("output"): 
    os.makedirs("output")
ml_colorise.test_video("input/vid_short.mp4","output/vid.mp4","tmp",ml_colorise.ML_TYPE_DEEPAI,ml_colorise.ML_MODE_COLORIZE)