import numpy as np
from PIL import Image

def display_one_image(data, flattend = False, save = False, file_name = None):
    if flattend:
        im = Image.fromarray(data.reshape((48, 48)))
    im = Image.fromarray(data)
    im.show()
    if save:
        im.save("data/"+file_name+".jpg")

    return