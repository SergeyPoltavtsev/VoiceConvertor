from scipy.misc import imread, imresize, imsave
from datetime import datetime
import numpy as np
import os


# import argparse
# from models import VGG16

# Not needed
# mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
# def add_mean(img):
#     for i in range(3):
#         img[0,:,:,i] += mean[i]
#     return img

# def sub_mean(img):
#     for i in range(3):
#         img[0,:,:,i] -= mean[i]
#     return img

def read_image(path):
    """
    Reads and processes an image from a specified path
    
    Inputs:
    - path: A path to an image
    
    Returns: 
    3-D matrix of RGB values
    """
    img = imread(path, mode='RGB')
    # Resize if ratio is specified
    img = imresize(img, (224, 224))
    img = process_image(img)
    return img


def process_image(arr):
    arr = arr.astype(np.float32)
    arr = arr[None, ...]
    # Subtract the image mean
    # arr = sub_mean(arr)
    return arr


def save_image(im, iteration, out_dir):
    img = im.copy()
    # Add the image mean
    # img = add_mean(img)
    img = np.clip(img[0, ...], 0, 255).astype(np.uint8)
    nowtime = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    imsave("{}/neural_art_{}_iteration{}.png".format(out_dir, nowtime, iteration), img)
