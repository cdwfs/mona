""""""""""""""""""""""""""""""""""""""""""""""""""""" 
|                                                   |
| evo.py - pyCuda shell for running Cuda kernels    |
|                                                   |
|                                                   |
|                                                   |
|  CUDA Parallel Particle Swarm Optimized EvoLisa   |
|            CS264 Final Project 2009               |
|            by Drew Robb & Joy Ding                |
|                                                   |
"""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as N
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math, random
from jinja2 import Template
import bmp
from PIL import Image
from PIL import _imaging
import os, sys

def im2array(im):
    a = N.fromstring(im.tostring(), N.uint8)
    a.shape = im.size[1], im.size[0], 3
    return a

def array2im(a):
    mode = "RGBA"
    return PIL.Image.fromstring(mode, (a.shape[1], a.shape[0]), a.tostring())

def image2array(path):
    return im2array(PIL.Image.open(path))


# editable program parameters
iters = 4000          # total number of iterations
psoiters = 1000       # maximum number of iterations at each PSO stage (overridden by checklimit)
dim = 800             # number of triangles to render (max)
size = 30             # number of particles for PSO
nhoodsize = 12        # size of neighboor hood (PSO topology)
ksize = 16            # size of block in threads (shouldn't be changed)
alphalimit = 0.2;    # maximum allowed alpha value (1 = total white/negative allowed)
checklimit = 150      # terminate PSO particle after this many static global best updates 
scale = 2             # increase output image dimensions by this factor
outputperiters = 100  # output image after this many optimizations

# clear temp directory
os.system("rm ./temp_images/*")

# setup input image
path = sys.argv[1]    # input image
oim = im2array(Image.open(path)) / 255.0
width = oim.shape[1]
height = oim.shape[0]
assert width % 4 == 0, 'Weird bug of pycuda texture, use image with width divisible by 4'
oim = N.concatenate((oim, N.ones((height,width, 1))), axis = 2).astype(N.float32)
oim = N.ravel(oim).astype(N.float32)
Goim = cuda.to_device(oim)

# render CUDA source, and setup texture of original image
tpl = Template(open("evo.cu","r").read(), line_statement_prefix="////")
rend = tpl.render(
    S = size,
    D = dim,
    nhoodsize = nhoodsize,
    W = width,
    H = height,
    KSIZE = ksize,
    psoiters = psoiters,
    ALPHALIM = alphalimit,
    ALPHAOFFSET = alphalimit/2,
    CHECKLIM = checklimit,
    SCALE = scale)
cudacmp = SourceModule(rend)
Grunfunc = cudacmp.get_function("run")
Grenderfunc = cudacmp.get_function("render")
Grenderprooffunc = cudacmp.get_function("renderproof")
Currimg = cudacmp.get_texref("currimg")
Refimg = cudacmp.get_texref("refimg")
descr = pycuda.driver.ArrayDescriptor()
descr.format = pycuda.driver.array_format.FLOAT
descr.height = height
descr.width  = width
descr.num_channels = 4
Refimg.set_address_2d(Goim, descr, 4*4*width)


# vector of solution triangles
curr = N.zeros((dim, 10)).astype(N.float32)
Gcurr = cuda.to_device(curr)
oldcurr = curr

# array to hold rendered solution on GPU
im = N.ones((height, width, 4)).astype(N.float32)* N.array([0,0,0,1]).astype(N.float32)
im = N.ravel(im).astype(N.float32)
Gim = cuda.to_device(im)

# array to hold scaled up solution, for output
im3 = N.ones((scale*height, scale*width, 4)).astype(N.float32)* N.array([0,0,0,1]).astype(N.float32)
Gim3 = cuda.to_device(N.ravel(im3))

# holds index of triangle we are updating
K = N.ones(1).astype(N.int32)

# current score of this iterations
score = N.array(1000000000.0).astype(N.float32)

Gscore = cuda.to_device(score)
bestscore = 10000000000000.0 

# Main loop
for i in range(iters):
    
    # get current solution
    curr = cuda.from_device_like(Gcurr, curr)
    
    # chooce new random triangle to update
    K = N.random.rand(1) * min(i/2, dim)
    K = K.astype(N.int32)
    GK = cuda.to_device(K)

    # render initial solution
    Grenderfunc(Gim, Gcurr, GK, Gscore, block=(ksize,ksize,1), grid=(1,1))
    Currimg.set_address_2d(Gim, descr, 4*4*width)
    score = cuda.from_device_like(Gscore, score)

    # check that this isn't a regression, revert and pick new K if so
    if score * (1.0 - 2.0 / dim) > bestscore:
        curr = oldcurr
        Gcurr = cuda.to_device(curr)
        K = N.random.rand(1) * min(i/2, dim)
        K = K.astype(N.int32)
        GK = cuda.to_device(K)
        Grenderfunc(Gim, Gcurr, GK, Gscore, block=(ksize,ksize,1), grid=(1,1))
        score = cuda.from_device_like(Gscore, score)
    # texturize current solution
    Currimg.set_address_2d(Gim, descr, 4*4*width)

    # create random data for this PSO iter, and send to device
    pos = N.random.rand(size, 10).astype(N.float32)               # position
    vel = (N.random.rand(size, 10).astype(N.float32) - 0.5) / 2   # velosity
    fit =  N.zeros(size).astype(N.float32) + 100000000000000.0    # fitness
    lbest = N.random.rand(size, 10).astype(N.float32)             # local best position
    lbval = N.zeros(size).astype(N.float32) + 100000000000000.0   # local best fitness
    nbest = N.random.rand(size, 10).astype(N.float32)             # neighborhood best position
    nbval = N.zeros(size).astype(N.float32) + 100000000000000.0   # neighborhood best fitness
    gbval = N.array(1000000000.0).astype(N.float32)               # global best value
    Gpos = cuda.to_device(pos)
    Gvel = cuda.to_device(vel)
    Gfit = cuda.to_device(fit)
    Glbest = cuda.to_device(lbest)
    Glbval = cuda.to_device(lbval)
    Gnbest = cuda.to_device(nbest)
    Gnbval = cuda.to_device(nbval)
    Ggbval = cuda.to_device(gbval)
       
    # run the pso kernel! the big one!
    Grunfunc(Gcurr, Gpos, Gvel, Gfit, Glbest, Glbval, Gnbest, Gnbval, Ggbval, GK, block=(ksize,ksize,1), grid=(size,1), texrefs=[Refimg, Currimg])

    # update best score if needed
    if score < bestscore and score != 0:
        bestscore = score

    # print output
    print score, i

    # update solution (tentatively)
    oldcurr = curr

    #visual output
    if i % outputperiters == 0:
        K = -N.ones(1).astype(N.int32)
        GK = cuda.to_device(K)
        Grenderprooffunc(Gim3, Gcurr, block=(ksize,ksize,1), grid=(1,1))
        im2 = 255.0 * cuda.from_device_like(Gim3, im3)
        im2 = im2.astype(N.uint8)
        X = array2im(im2)
        X.save("./temp_images/%s%s.jpg" % (path, iters), "JPEG", quality=100)



