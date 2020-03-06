import ml_colorise
import image_convert

# IMAGE ML
image_convert.video_to_images("input/vid.mp4","input")
ml_colorise.test("input","output",ml_colorise.ML_TYPE_TF,ml_colorise.ML_MODE_COLORIZE)
image_convert.images_to_video("output",output_file="vid_colour.avi")
ml_colorise.test("output","output_super",ml_colorise.ML_TYPE_DEEPAI,ml_colorise.ML_MODE_SUPER_RES)
image_convert.images_to_video("output_super",output_file="vid_colour_super.avi")