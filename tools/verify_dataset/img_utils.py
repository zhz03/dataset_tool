# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
from PIL import Image

class ImageLoader():
    """
    Class to handle image loading and processing.
    This class provides methods to load images, decode camera parameters, and get the camera intrinsic matrix.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_K(image_w, image_h, fov):
        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        """
        Get the camera intrinsic matrix K from the image width, image height, and field of view (fov).
        For example:     
            image_w = 1920
            image_h = 1080
            fov = 120
        """

        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0
        return K

    @staticmethod
    def decode_wh(K):
        """
        Extracts the image width and height from the camera intrinsic matrix K.

        The intrinsic matrix K is assumed to have the following structure:
            K = [[Fx,  0, image_w / 2],
                [ 0, Fy, image_h / 2],
                [ 0,  0,          1]]

        :param K: A 3x3 camera intrinsic matrix (numpy array or list of lists).
        :return: A tuple containing (image_w, image_h) as integers.
        """
        # Convert K to a numpy array if it's not already
        K = np.array(K)
        
        # Validate the shape of K
        if K.shape != (3, 3):
            raise ValueError(f"Intrinsic matrix K must be of shape (3, 3), but got {K.shape}")
        
        # Extract the (0, 2) and (1, 2) elements which correspond to image_w/2 and image_h/2 respectively
        image_w_half = K[0, 2]
        image_h_half = K[1, 2]
        
        # Compute the full image width and height
        image_w = 2 * image_w_half
        image_h = 2 * image_h_half
        
        # Optionally, round the values to the nearest integer
        image_w = int(round(image_w))
        image_h = int(round(image_h))
        
        return image_w, image_h

    @staticmethod
    def process_jpeg_to_array(file_path):
        """
        Reads a JPEG file and converts it into a numpy array with the correct RGB channel order.

        :param file_path: Path to the JPEG image file.
        :return: Numpy array of shape (height, width, 3) in correct RGB order.
        """
        # Open the image using PIL
        image = Image.open(file_path)

        # Convert the image to a numpy array (shape: (height, width, 3) in RGB)
        im_array = np.array(image, dtype=np.uint8)

        # Ensure the array has 3 channels (RGB). If there are 4 channels (RGBA), drop the alpha channel.
        if im_array.shape[-1] == 4:
            im_array = im_array[:, :, :3]  # Retain only the RGB channels

        # Return the array in its original RGB order without reversing channels
        return im_array