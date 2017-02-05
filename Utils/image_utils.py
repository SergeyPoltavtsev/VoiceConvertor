from scipy.misc import imread, imresize, imsave
from datetime import datetime
import numpy as np
import os


def read_image(path):
    """
    Reads and processes an image from a specified path.

    :param path: A path to an image
    :return: 3-D matrix of RGB values
    """

    img = imread(path, mode='RGB')
    # Resize if ratio is specified
    img = imresize(img, (224, 224))
    img = process_image(img)
    return img


def process_image(img):
    """
    Image preprocessing which depends on the task.
    Casts to the float32 and adds one more dimension.

    :param img: An image to process
    :return: Processed image
    """

    img = img.astype(np.float32)
    img = img[None, ...]
    # Subtract the image mean
    # img = sub_mean(img)
    return img


def save_image(image, iteration, out_dir):
    """
    Saves an image to the disk

    :param image: An image to save
    :param iteration: Iteration number
    :param out_dir: output directory
    :return:
    """

    img = image.copy()
    # Add the image mean
    # img = add_mean(img)
    img = np.clip(img[0, ...], 0, 255).astype(np.uint8)
    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    imsave("{}/neural_art_{}_iteration{}.png".format(out_dir, now, iteration), img)
