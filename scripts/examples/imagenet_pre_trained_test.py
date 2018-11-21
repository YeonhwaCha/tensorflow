from PIL import Image

import cv2
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from scripts.environments import global_env
from scripts.preprocessing import preprocessing
from scripts.networks import networks

LABEL_PATH = os.path.join(global_env.DATASET_DIR, 'imagenet1000_labels.txt')
LOG_DIR    = os.path.join(global_env.DATASET_DIR, 'Log/test')
IMAGE_PATH = os.path.join(global_env.DATASET_DIR, 'Data/test')

###############################################################
# pre-trained weight [1000/1001] :: None is considered or not
# -> pre-trained file이 none을 고려한 경우랑 아닌 경우로 저장되어져있음.
# check the name of mobilenet
# check the extension of nasnet mobile
# there is no files : resnet_v2_152, resnet_v2_200

if __name__ == "__main__":
    # No ckpt : [10]resnet_v2_152, [11]resnet_v2_200
    # ??????? : nasnet_mobile, nasnet_large, pnasnet_mobile, pnasnet_large
    network_name = ['inception_v1', 'inception_v2', 'inception_v3', 'inception_v4',
                    'inception_resnet_v2',
                    'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152',
                    'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'resnet_v2_200',
                    'vgg_16', 'vgg_19',
                    'mobilenet_v1', 'mobilenet_v2', 'mobilenet_v2_035', 'mobilenet_v2_140'
                    'nasnet_mobile', 'nasnet_large',
                    'pnasnet_mobile', 'pnasnet_large']
    idx = 1
    print('[Network_Name] -', network_name[idx])

    ###############################
    # INPUT
    ###############################
    # load dataset - imagenet
    filenames = [IMAGE_PATH + '/indigo_bird.jpg',
                 IMAGE_PATH + '/mug.jpg',
                 IMAGE_PATH + '/panda.jpg']
    labels = open(LABEL_PATH, 'r').read().split('\n')
    filename_queue = tf.train.string_input_producer(filenames)

    var_input = tf.placeholder(tf.float32, [None, None, 3], name='Input')
    var_preprocessing = tf.placeholder(tf.float32, [None, None, 3], name='Preprocess')

    ################################
    # Network
    ################################
    network = networks.Network(network_name[idx])
    resize = network['input_size']
    var_network = tf.placeholder(tf.float32, shape=[None] + list(network['input_size']), name='Network')
    arg_scope = network['arg_scope']()
    arg_scope = {**arg_scope}

    with tf.Graph().as_default():
        with tf.name_scope('input'):
            # original
            tf.summary.image('origin', tf.expand_dims(var_input, 0))

            # preprocessing
            preprocessing_obj = preprocessing.Preprocessing(network_name[idx])
            tf.summary.image('preprocessing', tf.expand_dims(var_preprocessing, 0))
            print(preprocessing_obj['description'])

        with tf.name_scope('network'):
            with slim.arg_scope(arg_scope):
                # network (ex. inception_v1)
                logits, end_points = network['net'](var_network,
                                                    num_classes=network['num_classes'],
                                                    is_training=False,
                                                    **network['kwargs'])
                predictions = tf.argmax(logits, 1)

    init = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()

    # GPU configuration
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session(config=gpu_config) as sess:
        sess.run(init)
        saver.restore(sess, network['ckpt_path'])
        summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(len(filenames)):  # length of your filename list
            value = filename_queue.dequeue()
            image = tf.read_file(value.eval())
            img_jpg = tf.image.decode_jpeg(image)  # use png or jpg decoder based on your files.
            input_img = img_jpg.eval()  # here is your image Tensor :)
            preprocessing_tensor = preprocessing_obj['preprocessing_for_test'](img_jpg, resize[0], resize[1])
            preprocessed_img = sess.run([preprocessing_tensor])
            prediction_result = sess.run(predictions[0], feed_dict={var_network: preprocessed_img})
            if(network['num_classes'] == 1001):
                print('[Prediction]', labels[prediction_result-1])
            else:
                print('[Prediction]', labels[prediction_result])
            summary = sess.run(summary_op, feed_dict={var_input: input_img, var_preprocessing: preprocessing_tensor.eval()})
            summary_writer.add_summary(summary)

        coord.request_stop()
        coord.join(threads)
