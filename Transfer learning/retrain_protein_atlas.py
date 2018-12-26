# This work is derived from the example code at:
# https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py,
# and modified to train the image classification problem for Human Protein Atlas.
# A tutorial for the original example code can be found at:
# https://www.tensorflow.org/hub/tutorials/image_retraining.

"""Simple transfer learning with image modules from TensorFlow Hub.

This example shows how to train an image classifier based on any
TensorFlow Hub module that computes image feature vectors. By default,
it uses the feature vectors computed by Inception V3 trained on ImageNet.
See https://github.com/tensorflow/hub/blob/master/docs/modules/image.md
for more options.

The top layer receives as input a 2048-dimensional vector (assuming
Inception V3) for each image. We train a FC layer followed by a 
softmax layer on top of this representation. The softmax layer 
contains N=28 labels.

```bash
python retrain.py --image_dir ~/flower_photos
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd 
import os, sys, datetime, io
import matplotlib.pyplot as plt 
import skimage
from skimage.transform import resize
from PIL import Image
import json
import scipy.io as sio
import random
from data_handler_transfer_learning import DataHandler

FLAGS = None

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'

# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')


def f1(y_true, y_pred):
  """Calculate the F1 score
  
  Args:
    y_true: the y-labels, dimension [batch_size, num_classes(=28)]
    y_pred: the final_tensor, in the range of (0,1), of dimension 
            [batch_size, num_classes(=28)]
  
  Returns: 
    the macro F1 score, which is the F1 scores averaged over all the classes
  """
  y_pred = tf.round(y_pred)

  tp = tf.reduce_sum(tf.cast(y_true*y_pred, tf.float32), axis=0)
  tn = tf.reduce_sum(tf.cast((1-y_true)*(1-y_pred), tf.float32), axis=0)
  fp = tf.reduce_sum(tf.cast((1-y_true)*y_pred, tf.float32), axis=0)
  fn = tf.reduce_sum(tf.cast(y_true*(1-y_pred), tf.float32), axis=0)

  precision = tp / (tp + fp + tf.keras.backend.epsilon())
  recall = tp / (tp + fn + tf.keras.backend.epsilon())

  f1 = 2*precision*recall / (precision+recall+tf.keras.backend.epsilon())
  f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
  return tf.reduce_mean(f1)

def f1_loss(y_true, y_pred):
  """Calculate the F1 loss based on F1 score 
  In calculating the F1 loss, do not round y_pred to ensure that 
  the f1_loss is differentiable with respect to y_pred

  Args:
    y_true: the y-labels, dimension [batch_size, num_classes(=28)]
    y_pred: the final_tensor, in the range of (0,1), of dimension 
            [batch_size, num_classes(=28)]
  
  Returns: 
    the macro F1 score, which is the F1 scores averaged over all the classes
  """
  tp = tf.reduce_sum(tf.cast(y_true*y_pred, tf.float32), axis=0)
  tn = tf.reduce_sum(tf.cast((1-y_true)*(1-y_pred), tf.float32), axis=0)
  fp = tf.reduce_sum(tf.cast((1-y_true)*y_pred, tf.float32), axis=0)
  fn = tf.reduce_sum(tf.cast(y_true*(1-y_pred), tf.float32), axis=0)

  precision = tp / (tp + fp + tf.keras.backend.epsilon())
  recall = tp / (tp + fn + tf.keras.backend.epsilon())

  f1 = 2*precision*recall / (precision+recall+tf.keras.backend.epsilon())
  f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
  return 1 - tf.reduce_mean(f1)

# The following function is not used to create the metadata
def create_image_lists(image_dir, testing_percentage, validation_percentage):
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    An OrderedDict containing an entry for each label subfolder, with images
    split into training, testing, and validation sets within each label.
    The order of items defines the class indices.
  """
  if not tf.gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None
  result = collections.OrderedDict()
  sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = sorted(set(os.path.normcase(ext)  # Smash case on Windows.
                            for ext in ['JPEG', 'JPG', 'jpeg', 'jpg']))
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(tf.gfile.Glob(file_glob))
    if not file_list:
      tf.logging.warning('No files found')
      continue
    if len(file_list) < 20:
      tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      tf.logging.warning(
          'WARNING: Folder {} has more than {} images. Some images will '
          'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result

def get_image_path(image_lists, label_name, index, image_dir, category):
  """Returns a path to an image for a label at the given index.

  Args:
    image_lists: OrderedDict of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
  base_name = image_lists[index]['path'].split('/')[-1]

  full_path = os.path.join(image_dir, label_name, base_name)

  return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, module_name):
  """Returns a path to a bottleneck file for a label at the given index.

  Args:
    image_lists: OrderedDict of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.
    module_name: The name of the image module being used.

  Returns:
    File system path string to an image that meets the requested parameters.
  """

  module_name = (module_name.replace('://', '~')  # URL scheme.
                 .replace('/', '~')  # URL and Unix paths.
                 .replace(':', '~').replace('\\', '~'))  # Windows paths.
  return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '_' + module_name + '.txt'

def create_module_graph(module_spec):
  """Creates a graph and loads Hub Module into it.

  Args:
    module_spec: the hub.ModuleSpec for the image module being used.

  Returns:
    graph: the tf.Graph that was created.
    bottleneck_tensor: the bottleneck values output by the module.
    resized_input_tensor: the input images, resized as expected by the module.
    wants_quantization: a boolean, whether the module has been instrumented
      with fake quantization ops.
  """
  height, width = hub.get_expected_image_size(module_spec)
  with tf.Graph().as_default() as graph:
    resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
    m = hub.Module(module_spec)
    bottleneck_tensor = m(resized_input_tensor)
    wants_quantization = any(node.op in FAKE_QUANT_OPS
                             for node in graph.as_graph_def().node)
  return graph, bottleneck_tensor, resized_input_tensor, wants_quantization


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    decoded_image_tensor: Output of initial image resizing and preprocessing.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
  # First decode the JPEG image, resize it, and rescale the pixel values.
  resized_input_values = image_data

  bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor, dh):
  """Create a single bottleneck file."""
  tf.logging.info('Creating bottleneck at ' + bottleneck_path)
  image_path = image_lists[index]['path']

  image_data = dh.load_image(image_path)
  image_data = np.array([image_data])

  try:
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, decoded_image_tensor,
        resized_input_tensor, bottleneck_tensor)
  except Exception as e:
    raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                 str(e)))
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, module_name, dh):
  """Retrieves or calculates bottleneck values for an image.

  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: OrderedDict of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of which set to pull images from - training, testing,
    or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The output tensor for the bottleneck values.
    module_name: The name of the image module being used.

  Returns:
    Numpy array of values produced by the bottleneck layer for the image.
  """

  sub_dir = label_name
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category, module_name)

  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor, dh)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()

  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    tf.logging.warning('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, module_name,
                      validation_percentage, testing_percentage, dh):
  """Ensures all the training, testing, and validation bottlenecks are cached.

  Because we're likely to read the same image multiple times (if there are no
  distortions applied during training) it can speed things up a lot if we
  calculate the bottleneck layer values once for each image during
  preprocessing, and then just read those cached values repeatedly during
  training. Here we go through all the images we've found, calculate those
  values, and save them off.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: OrderedDict of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The penultimate output layer of the graph.
    module_name: The name of the image module being used.

  Returns:
    Nothing.
  """
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)

  for index, example in enumerate(image_lists):
    # get path and labels from the attributes of the example
    file_path = example['path']
    labels = example['labels']
    label_name = '_'.join([str(label) for label in labels])

    # get category: attribute the example to one of the categories based 
    # on the percentages, and then save the category as an attribute 
    # of the example (i.e. image_lists[index]['category'])
    hash_name = re.sub(r'_nohash_.*$', '', file_path)
    hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_IMAGES_PER_CLASS))
    if percentage_hash < validation_percentage:
      category = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
      category = 'testing'
    else:
      category = 'training'
    image_lists[index]['category'] = category
    
    get_or_create_bottleneck(
            sess, image_lists, label_name, index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, module_name, dh)

    how_many_bottlenecks += 1
    if how_many_bottlenecks % 1000 == 0:
      tf.logging.info(
          str(how_many_bottlenecks) + ' bottleneck files created.')

def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, module_name, class_count, dh):
  """Retrieves bottleneck values for cached images.

  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.

  Args:
    sess: Current TensorFlow Session.
    image_lists: OrderedDict of training images for each label.
    how_many: If positive, a random sample of this size will be chosen.
    If negative, all bottlenecks will be retrieved.
    category: Name string of which set to pull from - training, testing, or
    validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    image_dir: Root folder string of the subfolders containing the training
    images.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.
    module_name: The name of the image module being used.

  Returns:
    List of bottleneck arrays, their corresponding ground truths, and the
    relevant filenames.
  """
  bottlenecks = []
  ground_truths = []
  filenames = []
  # Get a batch of examples if the batch size is specified (how_many >= 0).
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      index = random.randrange(len(image_lists) )
      while image_lists[index]['category'] != category:
        index = random.randrange(len(image_lists) )

      # Retrieve the attributes of the example
      example = image_lists[index]
      file_path = example['path']
      labels = example['labels']
      temp_ground_truth = np.zeros(class_count)
      temp_ground_truth[labels] = 1
      label_name = '_'.join([str(label) for label in labels])      

      image_name = file_path
      bottleneck = get_or_create_bottleneck(
          sess, image_lists, label_name, index, image_dir, category,
          bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor, module_name, dh)
      bottlenecks.append(bottleneck)
      
      ground_truths.append(temp_ground_truth)
      filenames.append(image_name)

  # Return all the examples in the category if the batch size < 0.
  else:
    # Retrieve all bottlenecks.
    for index in range(len(image_lists)):
      if image_lists[index]['category'] == category:

        example = image_lists[index]
        file_path = example['path']
        labels = example['labels']
        temp_ground_truth = np.zeros(class_count)
        temp_ground_truth[labels] = 1
        label_name = '_'.join([str(label) for label in labels])      
        image_name = file_path

        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, module_name, dh)
        bottlenecks.append(bottleneck)
        
        ground_truths.append(temp_ground_truth)
        filenames.append(image_name)

  return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(
    sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
    distorted_image, resized_input_tensor, bottleneck_tensor):
  """Retrieves bottleneck values for training images, after distortions.

  If we're training with distortions like crops, scales, or flips, we have to
  recalculate the full model for every image, and so we can't use cached
  bottleneck values. Instead we find random images for the requested category,
  run them through the distortion graph, and then the full graph to get the
  bottleneck results for each.

  Args:
    sess: Current TensorFlow Session.
    image_lists: OrderedDict of training images for each label.
    how_many: The integer number of bottleneck values to return.
    category: Name string of which set of images to fetch - training, testing,
    or validation.
    image_dir: Root folder string of the subfolders containing the training
    images.
    input_jpeg_tensor: The input layer we feed the image data to.
    distorted_image: The output node of the distortion graph.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.

  Returns:
    List of bottleneck arrays and their corresponding ground truths.
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
    image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                category)
    if not tf.gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = tf.gfile.FastGFile(image_path, 'rb').read()
    # Note that we materialize the distorted_image_data as a numpy array before
    # sending running inference on the image. This involves 2 memory copies and
    # might be optimized in other implementations.
    distorted_image_data = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: distorted_image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    bottlenecks.append(bottleneck_values)
    ground_truths.append(label_index)
  return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
  """Whether any distortions are enabled, from the input flags.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.

  Returns:
    Boolean value indicating whether any distortions should be applied.
  """
  return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, module_spec):
  """Creates the operations to apply the specified distortions.

  During training it can help to improve the results if we run the images
  through simple distortions like crops, scales, and flips. These reflect the
  kind of variations we expect in the real world, and so can help train the
  model to cope with natural data more effectively. Here we take the supplied
  parameters and construct a network of operations to apply them to an image.

  Cropping
  ~~~~~~~~

  Cropping is done by placing a bounding box at a random position in the full
  image. The cropping parameter controls the size of that box relative to the
  input image. If it's zero, then the box is the same size as the input and no
  cropping is performed. If the value is 50%, then the crop box will be half the
  width and height of the input. In a diagram it looks like this:

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  Scaling
  ~~~~~~~

  Scaling is a lot like cropping, except that the bounding box is always
  centered and its size varies randomly within the given range. For example if
  the scale percentage is zero, then the bounding box is the same size as the
  input and no scaling is applied. If it's 50%, then the bounding box will be in
  a random range between half the width and height and full size.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.
    graph.
    module_spec: The hub.ModuleSpec for the image module being used.

  Returns:
    The jpeg input layer and the distorted result tensor.
  """
  input_height, input_width = hub.get_expected_image_size(module_spec)
  input_depth = hub.get_num_image_channels(module_spec)
  jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  # Convert from full range of uint8 to range [0,1] of float32.
  decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  margin_scale = 1.0 + (random_crop / 100.0)
  resize_scale = 1.0 + (random_scale / 100.0)
  margin_scale_value = tf.constant(margin_scale)
  resize_scale_value = tf.random_uniform(shape=[],
                                         minval=1.0,
                                         maxval=resize_scale)
  scale_value = tf.multiply(margin_scale_value, resize_scale_value)
  precrop_width = tf.multiply(scale_value, input_width)
  precrop_height = tf.multiply(scale_value, input_height)
  precrop_shape = tf.stack([precrop_height, precrop_width])
  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
  precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                              precrop_shape_as_int)
  precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
  cropped_image = tf.random_crop(precropped_image_3d,
                                 [input_height, input_width, input_depth])
  if flip_left_right:
    flipped_image = tf.image.random_flip_left_right(cropped_image)
  else:
    flipped_image = cropped_image
  brightness_min = 1.0 - (random_brightness / 100.0)
  brightness_max = 1.0 + (random_brightness / 100.0)
  brightness_value = tf.random_uniform(shape=[],
                                       minval=brightness_min,
                                       maxval=brightness_max)
  brightened_image = tf.multiply(flipped_image, brightness_value)
  distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
  return jpeg_data, distort_result


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor,
                          quantize_layer, is_training):
  """Adds a new softmax and fully-connected layer for training and eval.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://www.tensorflow.org/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
        recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.
    quantize_layer: Boolean, specifying whether the newly added layer should be
        instrumented for quantization with TF-Lite.
    is_training: Boolean, specifying whether the newly add layer is for training
        or eval.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
  batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
  assert batch_size is None, 'We want to work with arbitrary batch size.'
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[batch_size, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(
        tf.float32, [batch_size, class_count], name='GroundTruthInput') #

  # hyperparameter to tune the relative scale of L2 regularization loss
  l2_scale = 0.001

  # FC layer from layer size 2048 to 256, followed by dropout layer
  with tf.name_scope("dense1"):
    x = tf.layers.dense(inputs=bottleneck_input, units=256, activation=tf.nn.relu,
                         kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=l2_scale))
    x = tf.layers.dropout(inputs=x, rate=0.5, training=is_training)

  # Logits Layer: from layer size 256 to 28
  with tf.name_scope("logit"):
    logits = tf.layers.dense(inputs=x, units=28, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=l2_scale))

  # sigmoid function applied to the logits
  final_tensor = tf.nn.sigmoid(logits)

  # The tf.contrib.quantize functions rewrite the graph in place for
  # quantization. The imported model graph has already been rewritten, so upon
  # calling these rewrites, only the newly added final layer will be
  # transformed.
  if quantize_layer:
    if is_training:
      tf.contrib.quantize.create_training_graph()
    else:
      tf.contrib.quantize.create_eval_graph()

  tf.summary.histogram('activations', final_tensor)

  # If this is an eval graph, we don't need to add loss ops or an optimizer.
  if not is_training:
    return None, None, bottleneck_input, ground_truth_input, final_tensor

  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=ground_truth_input, logits=logits))
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  # F1 loss
  with tf.name_scope('f1_loss'):
    f1_loss_value = f1_loss(ground_truth_input, final_tensor)
  tf.summary.scalar('f1_loss_value', f1_loss_value)
  l2_loss = tf.losses.get_regularization_loss()
  f1_loss_value += l2_loss

  # Optimizer
  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) 
    train_step = optimizer.minimize(f1_loss_value)

  return (train_step, cross_entropy_mean, f1_loss_value, bottleneck_input, ground_truth_input,
          final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
  """Inserts the operations we need to evaluate the metric of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Tuple of (evaluation step, prediction).
  """

  with tf.name_scope('accuracy'):    
    f1_score = f1(ground_truth_tensor, result_tensor)
  tf.summary.scalar('f1_score', f1_score)
  return f1_score


def run_final_eval(train_session, module_spec, class_count, image_lists,
                   jpeg_data_tensor, decoded_image_tensor,
                   resized_image_tensor, bottleneck_tensor, dh):
  """Runs a final evaluation on an eval graph using the test data set.

  Args:
    train_session: Session for the train graph with the tensors below.
    module_spec: The hub.ModuleSpec for the image module being used.
    class_count: Number of classes
    image_lists: OrderedDict of training images for each label.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_image_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.
  """
  test_bottlenecks, test_ground_truth, test_filenames = (
      get_random_cached_bottlenecks(train_session, image_lists,
                                    FLAGS.test_batch_size,
                                    'testing', FLAGS.bottleneck_dir,
                                    FLAGS.image_dir, jpeg_data_tensor,
                                    decoded_image_tensor, resized_image_tensor,
                                    bottleneck_tensor, FLAGS.tfhub_module, 
                                    class_count, dh))

  (eval_session, _, bottleneck_input, ground_truth_input, evaluation_step) = build_eval_session(module_spec, class_count)
  test_accuracy = eval_session.run(evaluation_step,
                                    feed_dict={
                                        bottleneck_input: test_bottlenecks,
                                        ground_truth_input: test_ground_truth
                                    })

  tf.logging.info('Final test F1 = %.1f%% (N=%d)' %
                  (test_accuracy * 100, len(test_bottlenecks)))

def build_eval_session(module_spec, class_count):
  """Builds an restored eval session without train operations for exporting.

  Args:
    module_spec: The hub.ModuleSpec for the image module being used.
    class_count: Number of classes

  Returns:
    Eval session containing the restored eval graph.
    The bottleneck input, ground truth, eval step, and prediction tensors.
  """
  # If quantized, we need to create the correct eval graph for exporting.
  eval_graph, bottleneck_tensor, resized_input_tensor, wants_quantization = (
      create_module_graph(module_spec))

  eval_sess = tf.Session(graph=eval_graph)
  with eval_graph.as_default():
    # Add the new layer for exporting.
    (_, _, bottleneck_input,
     ground_truth_input, final_tensor) = add_final_retrain_ops(
         class_count, FLAGS.final_tensor_name, bottleneck_tensor,
         wants_quantization, is_training=False)

    # Now we need to restore the values from the training graph to the eval
    # graph.
    tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)

    evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)

  return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input,
          evaluation_step)


def save_graph_to_file(graph_file_name, module_spec, class_count):
  """Saves an graph to file, creating a valid quantized one if necessary."""
  sess, _, _, _, _ = build_eval_session(module_spec, class_count)
  graph = sess.graph

  output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])

  with tf.gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())


def prepare_file_system():
  # Set up the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    pass
    # tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  if FLAGS.intermediate_store_frequency > 0:
    ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
  return


def add_jpeg_decoding(module_spec):
  """Adds operations that perform JPEG decoding and resizing to the graph..

  Args:
    module_spec: The hub.ModuleSpec for the image module being used.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """
  input_height, input_width = hub.get_expected_image_size(module_spec)
  input_depth = hub.get_num_image_channels(module_spec)
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  # Convert from full range of uint8 to range [0,1] of float32.
  decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  return jpeg_data, resized_image


def export_model(module_spec, class_count, saved_model_dir):
  """Exports model for serving.

  Args:
    module_spec: The hub.ModuleSpec for the image module being used.
    class_count: The number of classes.
    saved_model_dir: Directory in which to save exported model and variables.
  """
  # The SavedModel should hold the eval graph.
  sess, in_image, _, _, _, _ = build_eval_session(module_spec, class_count)
  with sess.graph.as_default() as graph:
    tf.saved_model.simple_save(
        sess,
        saved_model_dir,
        inputs={'image': in_image},
        outputs={'prediction': graph.get_tensor_by_name('final_result:0')},
        legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op')
    )


def main(_):
  # Needed to make sure the logging output is visible.
  # See https://github.com/tensorflow/tensorflow/issues/3047
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.image_dir:
    tf.logging.error('Must set flag --image_dir.')
    return -1

  # Prepare necessary directories that can be used during training
  prepare_file_system()

  # Set up the metadata: filename-label match
  data = pd.read_csv(FLAGS.metadata_file)
  metadata = []
  for filename, labels in zip(data['Id'], data['Target'].str.split(' ')):
      metadata.append({
          'path': os.path.join(FLAGS.image_dir, filename),
          'labels': np.array([int(label) for label in labels]),
          'category': None
          })

  image_lists = metadata
  image_dims = [299, 299, 3]

  dh = DataHandler(metadata, image_dims, FLAGS.train_batch_size)
    
  # 28 classes of proteins 
  class_count = 28

  # See if the command-line flags mean we're applying any distortions.
  do_distort_images = should_distort_images(
      FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
      FLAGS.random_brightness)

  # Set up the pre-trained graph.
  module_spec = hub.load_module_spec(FLAGS.tfhub_module)
  graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (
      create_module_graph(module_spec))

  # TENSOR DIMENSIONS

  # resized_image_tensor = [None, height, width, 3]
  # bottleneck_tensor = [None, V]
  # bottleneck_input = [batch_size, V] = placeholder(default=bottleneck_tensor)
  # ground_truth_input = [batch_size, class_count]
  # final_tensor = [batch_size, class_count]
  # jpeg_data_tensor = tf.string, name='DecodeJPGInput'
  # decoded_image_tensor = [batch, height, width, channels]

  # Add the new layer that we'll be training.
  with graph.as_default():
    (train_step, cross_entropy, train_f1_loss, bottleneck_input,
     ground_truth_input, final_tensor) = add_final_retrain_ops(
         class_count, FLAGS.final_tensor_name, bottleneck_tensor,
         wants_quantization, is_training=True)

  with tf.Session(graph=graph) as sess:
    # Initialize all weights: for the module to their pretrained values,
    # and for the newly added retraining layer to random initial values.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

    if do_distort_images:
      # We will be applying distortions, so set up the operations we'll need.
      (distorted_jpeg_data_tensor,
       distorted_image_tensor) = add_input_distortions(
           FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
           FLAGS.random_brightness, module_spec)
    else:
      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.
      cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                        FLAGS.bottleneck_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor,
                        bottleneck_tensor, FLAGS.tfhub_module, 
                        FLAGS.validation_percentage, FLAGS.testing_percentage, dh)
    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', #'/train',
                                         sess.graph)

    validation_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')#'/validation')

    # Create a train saver that is used to restore values into an eval graph
    # when exporting models.
    train_saver = tf.train.Saver()

    train_loss_list = []
    train_f1_list = []
    eval_loss_list = []
    eval_f1_list = []

    try:
      # Run the training for as many cycles as requested on the command line.
      for i in range(FLAGS.how_many_training_steps):
        # Get a batch of input bottleneck values, either calculated fresh every
        # time with distortions applied, or from the cache stored on disk.
        if do_distort_images:
          (train_bottlenecks,
           train_ground_truth) = get_random_distorted_bottlenecks(
               sess, image_lists, FLAGS.train_batch_size, 'training',
               FLAGS.image_dir, distorted_jpeg_data_tensor,
               distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
        else:
          (train_bottlenecks,
           train_ground_truth, _) = get_random_cached_bottlenecks(
               sess, image_lists, FLAGS.train_batch_size, 'training',
               FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
               decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
               FLAGS.tfhub_module, class_count, dh)
      
        # Feed the bottlenecks and ground truth into the graph, and run a training
        # step. Capture training summaries for TensorBoard with the `merged` op.
        train_summary, _ = sess.run(
            [merged, train_step],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        train_writer.add_summary(train_summary, i)
        train_writer.flush()

        # Every so often, print out how well the graph is training.
        is_last_step = (i + 1 == FLAGS.how_many_training_steps)
        if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
          train_accuracy, cross_entropy_value, train_f1_loss_value = sess.run(
              [evaluation_step, cross_entropy, train_f1_loss],
              feed_dict={bottleneck_input: train_bottlenecks,
                         ground_truth_input: train_ground_truth})
          tf.logging.info('%s: Step %d: Train F1 score = %.1f%%' %
                          (datetime.datetime.now(), i, train_accuracy * 100))
          tf.logging.info('%s: Step %d: Cross entropy = %f' %
                          (datetime.datetime.now(), i, cross_entropy_value))
          # TODO: Make this use an eval graph, to avoid quantization
          # moving averages being updated by the validation set, though in
          # practice this makes a negligable difference.
          validation_bottlenecks, validation_ground_truth, _ = (
              get_random_cached_bottlenecks(
                  sess, image_lists, FLAGS.validation_batch_size, 'validation',
                  FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                  decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                  FLAGS.tfhub_module, class_count, dh))
          # Run a validation step and capture training summaries for TensorBoard
          # with the `merged` op.
          validation_summary, validation_accuracy, eval_f1_loss_value = sess.run(
              [merged, evaluation_step, train_f1_loss],
              feed_dict={bottleneck_input: validation_bottlenecks,
                         ground_truth_input: validation_ground_truth})
          validation_writer.add_summary(validation_summary, i)
          validation_writer.flush()
          tf.logging.info('%s: Step %d: Validation F1 = %.1f%% (N=%d)' %
                          (datetime.datetime.now(), i, validation_accuracy * 100,
                           len(validation_bottlenecks)))

          train_loss_list.append(train_f1_loss_value)
          train_f1_list.append(train_accuracy)
          eval_f1_list.append(validation_accuracy)
          eval_loss_list.append(eval_f1_loss_value)

        # Store intermediate results
        intermediate_frequency = FLAGS.intermediate_store_frequency

        if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
            and i > 0):
          # If we want to do an intermediate save, save a checkpoint of the train
          # graph, to restore into the eval graph.
          train_saver.save(sess, CHECKPOINT_NAME)
          intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                    'intermediate_' + str(i) + '.pb')
          tf.logging.info('Save intermediate result to : ' +
                          intermediate_file_name)
          save_graph_to_file(intermediate_file_name, module_spec,
                             class_count)

      # After training is complete, force one last save of the train checkpoint.
      train_saver.save(sess, CHECKPOINT_NAME)

      # We've completed all our training, so run a final test evaluation on
      # some new images we haven't used before.
      run_final_eval(sess, module_spec, class_count, image_lists,
                     jpeg_data_tensor, decoded_image_tensor, 
                     resized_image_tensor, bottleneck_tensor, dh)

      # Write out the trained graph and labels with the weights stored as
      # constants.
      tf.logging.info('Save final result to : ' + FLAGS.output_graph)
      if wants_quantization:
        tf.logging.info('The model is instrumented for quantization with TF-Lite')

    except KeyboardInterrupt:
      tf.logging.info('Received KeyboardInterrupt - saving model')
      train_saver.save(sess, os.path.join(FLAGS.summaries_dir, 'model.ckpt'))


    file_name_mat = os.path.join(FLAGS.summaries_dir, 'training_results.mat')
    sio.savemat(file_name_mat, {'loss': np.array(train_loss_list),
                                 'train_f1': np.array(train_f1_list),
                                 'eval_f1': np.array(eval_f1_list)})

    file_name_txt = os.path.join(FLAGS.summaries_dir, 'training_results.txt')
    with open(file_name_txt, 'w') as out_file:
        out_file.write('loss: '+str(train_loss_list)+'\n')
        out_file.write('train_f1 :'+str(train_f1_list)+'\n')
        out_file.write('eval_f1:'+str(eval_f1_list)+'\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--metadata_file',
      type=str,
      default='../data/train.csv',
      help='Path to the metadata file (csv) containing training labels.'
  )

  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='/tmp/output_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='/tmp/intermediate_graph/',
      help='Where to save the intermediate graphs.'
  )
  parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='/tmp/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs/13',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=10000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.0003,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=200,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=200,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=False,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
  )
  parser.add_argument(
      '--tfhub_module',
      type=str,
      default=(
          'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'),
      help="""\
      Which TensorFlow Hub module to use.
      See https://github.com/tensorflow/hub/blob/master/docs/modules/image.md
      for some publicly available ones.\
      """)
  parser.add_argument(
      '--saved_model_dir',
      type=str,
      default='',
      help='Where to save the exported graph.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
