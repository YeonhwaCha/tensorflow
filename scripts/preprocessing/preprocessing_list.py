from models.research.slim.preprocessing import preprocessing_factory

####################################
# THE FORMAT OF PREPROCESSING LIST
# PREPROCESSING_LIST[name] = {
#   'preprocessing'   :
# }

PREPROCESSING_LIST = {}

name = 'inception_v1'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : inception\n'
                    '1) resize_images,\n' 
                    '2) randomly flip the image horizontally,\n'
                    '3) distort_color (fast_mode = True / False ),\n'
                    '4) distorted_image - 0.5,\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'inception_v2'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : inception\n'
                    '1) resize_images,\n' 
                    '2) randomly flip the image horizontally,\n'
                    '3) distort_color (fast_mode = True / False ),\n'
                    '4) distorted_image - 0.5,\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'inception_v3'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : inception\n'
                    '1) resize_images,\n' 
                    '2) randomly flip the image horizontally,\n'
                    '3) distort_color (fast_mode = True / False ),\n'
                    '4) distorted_image - 0.5,\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'inception_v4'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : inception\n'
                    '1) resize_images,\n' 
                    '2) randomly flip the image horizontally,\n'
                    '3) distort_color (fast_mode = True / False ),\n'
                    '4) distorted_image - 0.5,\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'inception_resnet_v2'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : inception\n'
                    '1) resize_images,\n' 
                    '2) randomly flip the image horizontally,\n'
                    '3) distort_color (fast_mode = True / False ),\n'
                    '4) distorted_image - 0.5,\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'resnet_v1_50'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : vgg\n'
                    '1) aspect_preserving_resize,\n' 
                    '2) central_crop,\n'
                    '3) change to float,\n'
                    '4) mean_image_subtraction : \n'
                    '   means = [R_MEAN=123.68, G_MEAN=116.78, B_MEAN=103.94]\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'resnet_v1_101'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : vgg\\n'
                    '1) aspect_preserving_resize,\n' 
                    '2) central_crop,\n'
                    '3) change to float,\n'
                    '4) mean_image_subtraction : \n'
                    '   means = [R_MEAN=123.68, G_MEAN=116.78, B_MEAN=103.94]\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'resnet_v1_152'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : vgg\\n'
                    '1) aspect_preserving_resize,\n' 
                    '2) central_crop,\n'
                    '3) change to float,\n'
                    '4) mean_image_subtraction : \n'
                    '   means = [R_MEAN=123.68, G_MEAN=116.78, B_MEAN=103.94]\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'resnet_v2_50'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : vgg\\n'
                    '1) aspect_preserving_resize,\n' 
                    '2) central_crop,\n'
                    '3) change to float,\n'
                    '4) mean_image_subtraction : \n'
                    '   means = [R_MEAN=123.68, G_MEAN=116.78, B_MEAN=103.94]\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'resnet_v2_101'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : vgg\\n'
                    '1) aspect_preserving_resize,\n' 
                    '2) central_crop,\n'
                    '3) change to float,\n'
                    '4) mean_image_subtraction : \n'
                    '   means = [R_MEAN=123.68, G_MEAN=116.78, B_MEAN=103.94]\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'resnet_v2_152'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : vgg\\n'
                    '1) aspect_preserving_resize,\n' 
                    '2) central_crop,\n'
                    '3) change to float,\n'
                    '4) mean_image_subtraction : \n'
                    '   means = [R_MEAN=123.68, G_MEAN=116.78, B_MEAN=103.94]\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'resnet_v2_200'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : vgg\\n'
                    '1) aspect_preserving_resize,\n' 
                    '2) central_crop,\n'
                    '3) change to float,\n'
                    '4) mean_image_subtraction : \n'
                    '   means = [R_MEAN=123.68, G_MEAN=116.78, B_MEAN=103.94]\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'vgg_16'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : vgg\\n'
                    '1) aspect_preserving_resize,\n' 
                    '2) central_crop,\n'
                    '3) change to float,\n'
                    '4) mean_image_subtraction : \n'
                    '   means = [R_MEAN=123.68, G_MEAN=116.78, B_MEAN=103.94]\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'vgg_19'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : vgg\\n'
                    '1) aspect_preserving_resize,\n' 
                    '2) central_crop,\n'
                    '3) change to float,\n'
                    '4) mean_image_subtraction : \n'
                    '   means = [R_MEAN=123.68, G_MEAN=116.78, B_MEAN=103.94]\n'
                    '5) distorted_image * 2.0,\n'
}

name = 'nasnet_mobile'
preprocessing_train_fn = preprocessing_factory.get_preprocessing(name, is_training=True)
preprocessing_test_fn  = preprocessing_factory.get_preprocessing(name, is_training=False)
PREPROCESSING_LIST[name] = {
    'preprocessing_for_train' : preprocessing_train_fn,
    'preprocessing_for_test'  : preprocessing_test_fn,
    'description' : '[Pre-processing] : inception\n'
                    '1) resize_images,\n' 
                    '2) randomly flip the image horizontally,\n'
                    '3) distort_color (fast_mode = True / False ),\n'
                    '4) distorted_image - 0.5,\n'
                    '5) distorted_image * 2.0,\n'
}