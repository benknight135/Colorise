import ml_colorise
import image_convert

# IMAGE ML
ml_colorise.test("input","output",ml_colorise.ML_TYPE_DEEPAI,ml_colorise.ML_MODE_COLORIZE)
image_convert.images_to_video("output","vid_colour.avi")