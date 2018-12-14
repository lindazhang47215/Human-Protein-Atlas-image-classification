import numpy as np 
import pandas as pd 
import os, sys, datetime, io
import matplotlib.pyplot as plt 
import skimage
from skimage.transform import resize
from PIL import Image
from cnn_model_small_tb_inference import cnn_model_small
from data_handler_mem_inference import DataHandler 
import json
import logging
import tensorflow as tf
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
from sklearn.metrics import f1_score
import scipy.io as sio

def f1(y_true, y_pred):
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
    
    tp = tf.reduce_sum(tf.cast(y_true*y_pred, tf.float32), axis=0)
    tn = tf.reduce_sum(tf.cast((1-y_true)*(1-y_pred), tf.float32), axis=0)
    fp = tf.reduce_sum(tf.cast((1-y_true)*y_pred, tf.float32), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true*(1-y_pred), tf.float32), axis=0)

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2*precision*recall / (precision+recall+tf.keras.backend.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - tf.reduce_mean(f1)


def train(params, model_dir):

    # expand parameters
    metadata_path = params['metadata_path']
    path_to_train = params['path_to_train']
    num_channels = params['num_channels']
    num_labels = params['num_labels']
    num_blocks = params['num_blocks']

    image_dims = params['image_dims']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']
    epsilon = params['epsilon'] 
    eval_fraction = params['eval_fraction']

    mode = 'train'

    # Set up the metadata -- filename-label match
    data = pd.read_csv(metadata_path)
    metadata = []
    for filename, labels in zip(data['Id'], data['Target'].str.split(' ')):
        metadata.append({
            'path': os.path.join(path_to_train, filename),
            'labels': np.array([int(label) for label in labels])
            })
    metadata = np.array(metadata)

    train_metadata = metadata[:int(metadata.shape[0]*(1.0-eval_fraction))]
    eval_metadata = metadata[int(-metadata.shape[0]*eval_fraction):]
    print(metadata.shape)
    print(train_metadata.shape)
    print(eval_metadata.shape)
    block_size = train_metadata.shape[0] // num_blocks
    print(block_size)

    # Load eval data (keep eval fraction ~0.05 so the total memory required is ~15G )
    eval_dh = DataHandler(eval_metadata, image_dims, batch_size)
    logger.info('Eval DH loaded.')

    # setting up the directory and dump the training params in the same folder
    time_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    directory = os.path.join('results',mode+'-'+time_now)
    if not os.path.exists(directory):
        os.makedirs(directory)

    param_file = os.path.join(directory, 'params.json')
    with io.open(param_file, 'w') as out_file:
        json.dump(params, out_file, indent = 4)


    # Setting up the log file
    log_file_handler = logging.FileHandler(os.path.join(directory, 'traininglog.log'))
    format_file_handler = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    log_file_handler.setFormatter(format_file_handler)
    logging.getLogger('').addHandler(log_file_handler)


    train_loss_list = []
    train_f1_list = []
    eval_loss_list = []
    eval_f1_list = []

    with tf.Session() as sess:
        logger.info('image size {0}'.format(image_dims))
        # set up the input and output variables
        images = tf.placeholder(name='images',
                                dtype=tf.float32,
                                shape=[None]+image_dims)
        labels = tf.placeholder(name='labels',
                                dtype=tf.float32,
                                shape=[None]+[num_labels])
        training_flag = tf.placeholder(name='training_flag', 
                                dtype = tf.bool)

        tf.summary.image('input', images, 4)
        # setting up the graph
        logits = cnn_model_small(images, training_flag)

        y_pred = tf.nn.softmax(logits, axis=1)

        # loss
        # cross entropy loss
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels = labels))
    
        # f1 loss
        loss = f1_loss(labels, y_pred)

        l2_loss = tf.losses.get_regularization_loss()
        loss += l2_loss
        tf.summary.scalar("loss", loss)

        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # calculate the f1_score
        f1_score = f1(labels, y_pred)
        tf.summary.scalar("f1_score", f1_score)

        summary_op = tf.summary.merge_all()

        # Setting up tensorboard support
        writer = tf.summary.FileWriter(os.path.join(directory,'logs/plot_1'), graph = sess.graph)#tf.get_default_graph())
        eval_writer = tf.summary.FileWriter(os.path.join(directory,'logs/plot_2'), graph = sess.graph)#tf.get_default_graph())

        # Initializer
        init = tf.global_variables_initializer()

        # Saver for the tensorflow model
        saver = tf.train.Saver()

        if model_dir is None:
            sess.run(init)
        else: 
            saver.restore(sess, os.path.join(model_dir, 'model.ckpt'))
        try: 
            for block in range(num_blocks):
                logger.info('Preparing data handler for block {0}...'.format(block))

                # Data handler 
                train_dh = DataHandler(train_metadata[block*block_size:(block+1)*block_size], image_dims, batch_size)
                
                train_batcher = train_dh.supply_random_batch(greenonly=False)
                eval_batcher = eval_dh.supply_random_batch(greenonly=False)
                
                logger.info('   Training DH loaded.')

                num_batches = eval_dh.size//batch_size + 1


                for epoch in range(num_epochs):
                    # logger.info('On epoch {0}'.format(epoch))
                    for batch in range(num_batches):
                        logger.info('On epoch {0} batch {1}'.format(epoch, batch))

                        # batch_images, batch_labels = train_dh.supply_batch(batch)
                        batch_images, batch_labels = next(train_batcher)
                        logger.info('  loaded batch'.format(batch))

                        if batch % 10 == 0:
                            eval_images, eval_labels = next(eval_batcher)

                            opt, train_summary, train_loss, train_f1_value = sess.run([optimizer, summary_op, loss, f1_score], 
                                                                                      feed_dict={images: batch_images,
                                                                                                 labels: batch_labels,
                                                                                                 training_flag: True})

                            eval_summary, eval_loss, eval_f1_value = sess.run([summary_op, loss, f1_score], 
                                                                              feed_dict={images: eval_images,
                                                                                         labels: eval_labels,
                                                                                         training_flag: False})
                            # write to summary
                            writer.add_summary(train_summary, block*num_epochs*num_batches+epoch*num_batches+batch)
                            writer.flush()
                            eval_writer.add_summary(eval_summary, block*num_epochs*num_batches+epoch*num_batches+batch)
                            eval_writer.flush()          

                            logger.debug('Labels {0}'.format(batch_labels))
                            logger.info('Train F1 score {0}'.format(train_f1_value))
                            logger.info('Train loss {0}'.format(train_loss))
                            logger.info('Eval F1 score {0}'.format(eval_f1_value))
                            logger.info('Eval loss {0}'.format(eval_loss))

                            train_loss_list.append(train_loss)
                            train_f1_list.append(train_f1_value)
                            eval_f1_list.append(eval_f1_value)
                            eval_loss_list.append(eval_loss)
                        else: 
                            sess.run(optimizer, 
                                    feed_dict={images: batch_images,
                                               labels: batch_labels,
                                               training_flag: True})

                saver.save(sess, os.path.join(directory, 'model.ckpt'))
        except KeyboardInterrupt:
            logger.info('Received KeyboardInterrupt - saving model')
            saver.save(sess, os.path.join(directory, 'model.ckpt'))

    # save as .mat file
    file_name = os.path.join(directory, 'training_results.mat')
    sio.savemat(file_name, {'loss': np.array(train_loss_list),
                                 'train_f1': np.array(train_f1_list),
                                 'eval_f1': np.array(eval_f1_list)})

    # save as .txt file
    file_name_txt = os.path.join(directory, 'training_results.txt')
    with open(file_name_txt, 'w') as out_file:
        out_file.write('loss: '+str(train_loss_list)+'\n')
        out_file.write('train_f1 :'+str(train_f1_list)+'\n')
        out_file.write('eval_f1:'+str(eval_f1_list)+'\n')


    print("done")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', default = None)
    args = parser.parse_args()

    params={}
    params['path_to_train']='data/train/'
    params['num_channels']= 4
    params['num_labels'] = 28
    params['num_blocks'] = 20
    params['metadata_path'] = 'data/train.csv'

    params['image_dims'] = [128,128,4]
    params['batch_size'] = 128
    params['num_epochs'] = 10
    params['learning_rate'] = 3e-5
    params['epsilon'] = 1e-7
    params['eval_fraction'] = 0.10
    train(params, args.modeldir)
