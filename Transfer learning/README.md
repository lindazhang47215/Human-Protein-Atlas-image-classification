This program is modified from retrain.py from https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py to work for the data structure and final layer architecture of our model. 

* To run the training, enter in the terminal

```
python retrain_protein_atlas.py --image_dir DIRECTORY_TO_TRAINING_IMAGES
```

The default metadata file location is '../data/train.csv', and the training images stored in the folder '../data/train'. The training images are not uploaded to github due to space constraint, but can be found at https://www.kaggle.com/c/human-protein-atlas-image-classification/data.

The code is modified to work for the way that the training images and labels are supplied and stored. We also modified the last few network layers and loss functions for the protein image classification application.