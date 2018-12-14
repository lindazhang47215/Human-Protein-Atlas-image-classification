This program is modified from retrain.py from https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py to work for the data structure and final layer architecture of our model. 

To run the training, enter in the terminal

python retrain_cell.py --image_dir DIRECTORY_TO_TRAINING_IMAGES

We modified the code to work for the way that our training images and labels are supplied and stored. We also modified the network layers and loss functions for our application.