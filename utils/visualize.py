import numpy as np
from PIL import Image

def display_one_image(data, save = False, file_name = None):

    im = Image.fromarray(data.reshape((48, 48)))
    im.show()
    if save:
        im.save("data/"+file_name+".jpg")

    return