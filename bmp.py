import numpy
import PIL
def im2array(im):
    a = numpy.fromstring(im.tostring(), numpy.uint8)
    a.shape = im.size[1], im.size[0], 3
    return a

def array2im(a):
    mode = "RGBA"
    return PIL.Image.fromstring(mode, (a.shape[1], a.shape[0]), a.tostring())

def image2array(path):
    return im2array(PIL.Image.open(path))

#im = Image.open("kb.jpg")
#im.rotate(45).show()
