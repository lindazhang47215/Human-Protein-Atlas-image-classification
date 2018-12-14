import numpy as np 
import pandas as pd 
# import seaborn as sb 
import os, sys
import matplotlib.pyplot as plt 
import skimage
from skimage.transform import resize
from PIL import Image
from cnn_model_small import cnn_model_small

PATH_TO_TRAIN='data/train/'
NUM_CHANNELS = 4
NUM_LABELS = 28

# def fill_targets(row):
#     row.Target = np.array(row.Target.split(" ")).astype(np.int)
#     for num in row.Target:
#         name = label_names[int(num)]
#         row.loc[name] = 1
#     return row
# def fill_targets(row):
#     row.Target = np.array(row.Target.split(" ")).astype(np.int)
#     for num in row.Target:
#         row.loc[label_names[int(num)]] = 1
#     return row

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
        
        image = np.stack((
            red_ch, 
            green_ch, 
            blue_ch, 
            yellow_ch), axis=-1)    

        image = resize(image, (self._image_dims[0], self._image_dims[1]), mode='reflect')

        return image

    def load_image_greenonly(self,path_to_image):
        red_ch = Image.open(path_to_image+'_red.png')
        green_ch = Image.open(path_to_image+'_green.png')
        blue_ch = Image.open(path_to_image+'_blue.png')
        yellow_ch = Image.open(path_to_image+'_yellow.png')
        
        # modify later
        red_ch += (np.array(yellow_ch)//2).astype(np.uint8) 
        blue_ch += (np.array(yellow_ch)//2).astype(np.uint8)

        image = np.array(green_ch)

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


    train_data = pd.read_csv('data/train.csv')
    train_metadata = []
    for filename, labels in zip(train_data['Id'], train_data['Target'].str.split(' ')):
        train_metadata.append({
            'path': os.path.join(PATH_TO_TRAIN, filename),
            'labels': np.array([int(label) for label in labels])
            })

    train_metadata = np.array(train_metadata)
    train_dh = DataHandler(train_metadata, [512,512,4], 4)
    image_0 = train_dh.load_image(train_metadata[0]['path'])
    print(image_0[:,100,0])
    plt.figure()
    plt.imshow(image_0)
    plt.show()


    # # Fetch a batch
    # train_batch = train_dh.supply_batch(greenonly=False)
    # images, labels = next(train_batch)

    # # Display the batch
    # fig, ax = plt.subplots(1,4,figsize=(20,5))

    # for i in range(4):
    #     ax[i].imshow(images[i])
    #     # ax[i].set_title(labels[i])
    #     # ax[i].set_facecolor('black')
    # plt.show()

    # model = cnn_model(images)


    # print('min: {0}, max: {1}'.format(images.min(), images.max()))
################################################################################################################


    # train_labels = pd.read_csv("data/train.csv")
    # # print(train_labels.head())
    # # print(train_labels.shape)

    # label_names = {
    #     0:  "Nucleoplasm",  
    #     1:  "Nuclear membrane",   
    #     2:  "Nucleoli",   
    #     3:  "Nucleoli fibrillar center",   
    #     4:  "Nuclear speckles",
    #     5:  "Nuclear bodies",   
    #     6:  "Endoplasmic reticulum",   
    #     7:  "Golgi apparatus",   
    #     8:  "Peroxisomes",   
    #     9:  "Endosomes",   
    #     10:  "Lysosomes",   
    #     11:  "Intermediate filaments",   
    #     12:  "Actin filaments",   
    #     13:  "Focal adhesion sites",   
    #     14:  "Microtubules",   
    #     15:  "Microtubule ends",   
    #     16:  "Cytokinetic bridge",   
    #     17:  "Mitotic spindle",   
    #     18:  "Microtubule organizing center",   
    #     19:  "Centrosome",   
    #     20:  "Lipid droplets",   
    #     21:  "Plasma membrane",   
    #     22:  "Cell junctions",   
    #     23:  "Mitochondria",   
    #     24:  "Aggresome",   
    #     25:  "Cytosol",   
    #     26:  "Cytoplasmic bodies",   
    #     27:  "Rods & rings"
    # }

    # for key, val in label_names.items():
    #     train_labels[val] = 0

    # train_labels = train_labels.apply(fill_targets, axis = 1)
    # print(train_labels.head())

    # # plot frequency

