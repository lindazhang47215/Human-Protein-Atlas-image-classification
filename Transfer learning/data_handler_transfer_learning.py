import numpy as np 
import pandas as pd 
import os, sys
import matplotlib.pyplot as plt 
import skimage
from skimage.transform import resize
from PIL import Image

PATH_TO_TRAIN='data/train/'
NUM_CHANNELS = 4
NUM_LABELS = 28

class DataHandler:
    def __init__(self, 
                 metadata,
                 image_dims,
                 batch_size):
        
        self._metadata = metadata
        self._image_dims = image_dims
        self._batch_size = batch_size

    @property
    def size(self):
        return self._metadata.shape[0]

    @property
    def metadata(self):
        return self._metadata

    def load_image(self,path_to_image):
        red_ch = Image.open(path_to_image+'_red.png')
        green_ch = Image.open(path_to_image+'_green.png')
        blue_ch = Image.open(path_to_image+'_blue.png')
        yellow_ch = Image.open(path_to_image+'_yellow.png')
        
        # modify later
        red_ch += (np.array(yellow_ch)//2).astype(np.uint8) 
        blue_ch += (np.array(yellow_ch)//2).astype(np.uint8)

        image = np.stack((
            red_ch, 
            green_ch, 
            blue_ch), axis=-1)    

        image = resize(image, (self._image_dims[0], self._image_dims[1]), mode='reflect')

        return image

    def supply_batch(self, greenonly):
        while True:
            indices = np.random.choice(len(self._metadata), self._batch_size)
            batch_images = np.empty((self._batch_size, self._image_dims[0], self._image_dims[1], self._image_dims[2]))
            batch_labels = np.zeros((self._batch_size, NUM_LABELS))

            for i, index in enumerate(indices):
                if greenonly: 
                    temp = self.load_image_greenonly(self._metadata[index]['path'])
                else:
                    temp = self.load_image(self._metadata[index]['path'])

                batch_images[i] = temp
                batch_labels[i][self._metadata[index]['labels']] = 1
            yield batch_images, batch_labels


if __name__ == "__main__":
    pass
