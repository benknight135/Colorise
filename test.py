import ml_colorise
import image_convert

# RESUME TEST
#ml_colorise.test("input","output",deepai.MODEL_COLORIZE,5134)
ml_colorise.test("input","output",ml_colorise.ML_TYPE_TF,ml_colorise.ML_MODE_COLORIZE,5134)
image_convert.images_to_video("output")
#deepai.test("output","output_super",deepai.MODEL_SUPER_RES,1058)
#image_convert.images_to_video("output_super")

'''
# FULL TEST
image_convert.video_to_images("input/vid.mp4","input")
deepai.test("input","output",deepai.MODEL_COLORIZE)
image_convert.images_to_video("output")
deepai.test("output","output_super",deepai.MODEL_SUPER_RES)
image_convert.images_to_video("output_super")
'''