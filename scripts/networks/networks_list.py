from scripts import networks

##############################
# THE FORMAT OF NETWORK LIST
# NETWORK_LIST[name] = {
#   'ckpt_path'   :
#   'frozen_path' :
#   'uff_path'    :
#   'input_size'  :
#   'num_classes' :
#   'net'         :
#   'arg_scope'   :
#   'kwargs'      :
#   'description' :
# }

NETWORK_LIST = {}

name = 'inception_v1'
from models.research.slim.nets.inception_v1 \
    import inception_v1 as net, \
        inception_v1_arg_scope as arg_scope
NETWORK_LIST[name] = {
    'ckpt_path'   : networks.CKPT_PATH.format(name),
    'num_classes' : 1001,
    'input_size'  : (224, 224, 3),
    'net'         : net,
    'arg_scope'   : arg_scope,
    'kwargs'      : {},
    'description' : '[Default]\n'
                    'inception_v1(inputs,\n' 
                    '             num_classes=1001,\n'
                    '             is_training=True,\n'
                    '             dropout_keep_prob=0.8,\n'
                    '             prediction_fn=slim.softmax,\n'
                    '             spatial_squeeze=True,\n'
                    '             reuse=None,\n'
                    '             scope="InceptionV1"\n'
                    '             global_pool=False)\n'
}

name = 'inception_v2'
from models.research.slim.nets.inception_v2 \
    import inception_v2 as net, \
        inception_v2_arg_scope as arg_scope
NETWORK_LIST[name] = {
    'ckpt_path'   : networks.CKPT_PATH.format(name),
    'num_classes' : 1001,
    'input_size'  : (224, 224, 3),
    'net'         : net,
    'arg_scope'   : arg_scope,
    'kwargs'      : {},
    'description' : '[Default]\n'
                    'inception_v2(inputs,\n' 
                    '             num_classes=1001,\n'
                    '             is_training=True,\n'
                    '             dropout_keep_prob=0.8,\n'
                    '             min_depth=16'
                    '             prediction_fn=slim.softmax,\n'
                    '             spatial_squeeze=True,\n'
                    '             reuse=None,\n'
                    '             scope="InceptionV2"\n'
                    '             global_pool=False)\n'
}

name = 'inception_v3'
from models.research.slim.nets.inception_v3 \
    import inception_v3 as net, \
        inception_v3_arg_scope as arg_scope
NETWORK_LIST[name] = {
    'ckpt_path'   : networks.CKPT_PATH.format(name),
    'num_classes' : 1001,
    'input_size'  : (224, 224, 3),
    'net'         : net,
    'arg_scope'   : arg_scope,
    'kwargs'      : {'create_aux_logits':False},
    'description' : 'inception_v3(inputs,\n' 
                    '             num_classes=1001,\n'
                    '             is_training=True,\n'
                    '             dropout_keep_prob=0.8,\n'
                    '             min_depth=16,\n'
                    '             depth_multiplier=1.0,\n'
                    '             prediction_fn=slim.softmax,\n'
                    '             spatial_squeeze=True,\n'
                    '             reuse=None,\n'
                    '             create_aux_logits=True,'
                    '             scope="InceptionV3"\n'
                    '             global_pool=False)\n'
}

name = 'inception_v4'
from models.research.slim.nets.inception_v4 \
    import inception_v4 as net, \
        inception_v4_arg_scope as arg_scope
NETWORK_LIST[name] = {
    'ckpt_path'   : networks.CKPT_PATH.format(name),
    'num_classes' : 1001,
    'input_size'  : (299, 299, 3),
    'net'         : net,
    'arg_scope'   : arg_scope,
    'kwargs'      : {},
    'description' : 'inception_v4(inputs,\n' 
                    '             num_classes=1001,\n'
                    '             is_training=True,\n'
                    '             dropout_keep_prob=0.8,\n'
                    '             reuse=None,\n'
                    '             scope="InceptionV4"\n'
                    '             create_aux_logits=True)\n'
}

name = 'inception_resnet_v2'
from models.research.slim.nets.inception_resnet_v2 \
    import inception_resnet_v2 as net, \
        inception_resnet_v2_arg_scope as arg_scope
NETWORK_LIST[name] = {
    'ckpt_path'   : networks.CKPT_PATH.format(name),
    'num_classes' : 1001,
    'input_size'  : (299, 299, 3),
    'net'         : net,
    'arg_scope'   : arg_scope,
    'kwargs'      : {'create_aux_logits': False},
    'description' : 'inception_resnet_v2(inputs,\n' 
                    '                    num_classes=1001,\n'
                    '                    is_training=True,\n'
                    '                    dropout_keep_prob=0.8,\n'
                    '                    reuse=None,\n'
                    '                    scope="InceptionResnetV2"\n'
                    '                    create_aux_logits=True,\n'
                    '                    activation_fn=tf.nn.relu)\n'
}

name = 'resnet_v1_50'
from models.research.slim.nets.resnet_v1 \
    import resnet_v1_50 as net, \
    resnet_arg_scope as arg_scope
NETWORK_LIST[name] = {
    'ckpt_path'   : networks.CKPT_PATH.format(name),
    'num_classes' : 1000,
    'input_size'  : (224, 224, 3),
    'net'         : net,
    'arg_scope'   : arg_scope,
    'kwargs'      : {},
    'description' : 'resnet_v1_50(inputs,\n' 
                    '             num_classes=1000,\n'
                    '             is_training=True,\n'
                    '             global_pool=True,\n'
                    '             output_stride=None,\n'
                    '             spatial_squeeze=True\n'
                    '             store_non_strided_activations=False,\n'
                    '             reuse=None,\n'
                    '             scope="resnet_v1_50")\n'
}

name = 'resnet_v1_101'
from models.research.slim.nets.resnet_v1 \
    import resnet_v1_101 as net, \
    resnet_arg_scope as arg_scope
NETWORK_LIST[name] = {
    'ckpt_path'   : networks.CKPT_PATH.format(name),
    'num_classes' : 1000,
    'input_size'  : (224, 224, 3),
    'net'         : net,
    'arg_scope'   : arg_scope,
    'kwargs'      : {},
    'description' : 'resnet_v1_101(inputs,\n' 
                    '              num_classes=1000,\n'
                    '              is_training=True,\n'
                    '              global_pool=True,\n'
                    '              output_stride=None,\n'
                    '              spatial_squeeze=True\n'
                    '              store_non_strided_activations=False,\n'
                    '              reuse=None,\n'
                    '              scope="resnet_v1_101")\n'
}

name = 'resnet_v1_152'
from models.research.slim.nets.resnet_v1 \
    import resnet_v1_152 as net, \
    resnet_arg_scope as arg_scope
NETWORK_LIST[name] = {
    'ckpt_path'   : networks.CKPT_PATH.format(name),
    'num_classes' : 1000,
    'input_size'  : (224, 224, 3),
    'net'         : net,
    'arg_scope'   : arg_scope,
    'kwargs'      : {},
    'description' : 'resnet_v1_101(inputs,\n' 
                    '              num_classes=1000,\n'
                    '              is_training=True,\n'
                    '              global_pool=True,\n'
                    '              output_stride=None,\n'
                    '              store_non_strided_activations=False,\n'
                    '              reuse=None,\n'
                    '              scope="resnet_v1_152")\n'
}

name = 'resnet_v2_50'
from models.research.slim.nets.resnet_v2 \
    import resnet_v2_50 as net, \
    resnet_arg_scope as arg_scope
NETWORK_LIST[name] = {
    'ckpt_path'   : networks.CKPT_PATH.format(name),
    'num_classes' : 1001,
    'input_size'  : (224, 224, 3),
    'net'         : net,
    'arg_scope'   : arg_scope,
    'kwargs'      : {},
    'description' : 'resnet_v2_50(inputs,\n' 
                    '             num_classes=1001,\n'
                    '             is_training=True,\n'
                    '             global_pool=True,\n'
                    '             output_stride=None,\n'
                    '             spatial_squeeze=True\n'
                    '             reuse=None,\n'
                    '             scope="resnet_v2_50")\n'
}

name = 'resnet_v2_101'
from models.research.slim.nets.resnet_v2 \
    import resnet_v2_101 as net, \
    resnet_arg_scope as arg_scope
NETWORK_LIST[name] = {
    'ckpt_path'   : networks.CKPT_PATH.format(name),
    'num_classes' : 1001,
    'input_size'  : (224, 224, 3),
    'net'         : net,
    'arg_scope'   : arg_scope,
    'kwargs'      : {},
    'description' : 'resnet_v2_101(inputs,\n' 
                    '              num_classes=1001,\n'
                    '              is_training=True,\n'
                    '              global_pool=True,\n'
                    '              output_stride=None,\n'
                    '              spatial_squeeze=True\n'
                    '              reuse=None,\n'
                    '              scope="resnet_v2_101")\n'
}

name = 'resnet_v2_152'
from models.research.slim.nets.resnet_v2 \
    import resnet_v2_152 as net, \
    resnet_arg_scope as arg_scope
NETWORK_LIST[name] = {
    'ckpt_path'   : networks.CKPT_PATH.format(name),
    'num_classes' : 1001,
    'input_size'  : (224, 224, 3),
    'net'         : net,
    'arg_scope'   : arg_scope,
    'kwargs'      : {},
    'description' : 'resnet_v2_152(inputs,\n' 
                    '              num_classes=1001,\n'
                    '              is_training=True,\n'
                    '              global_pool=True,\n'
                    '              output_stride=None,\n'
                    '              spatial_squeeze=True\n'
                    '              reuse=None,\n'
                    '              scope="resnet_v2_152")\n'
}

name = 'resnet_v2_200'
from models.research.slim.nets.resnet_v2 \
    import resnet_v2_200 as net, \
    resnet_arg_scope as arg_scope
NETWORK_LIST[name] = {
    'ckpt_path'   : networks.CKPT_PATH.format(name),
    'num_classes' : 1001,
    'input_size'  : (224, 224, 3),
    'net'         : net,
    'arg_scope'   : arg_scope,
    'kwargs'      : {},
    'description' : 'resnet_v2_200(inputs,\n' 
                    '              num_classes=1001,\n'
                    '              is_training=True,\n'
                    '              global_pool=True,\n'
                    '              output_stride=None,\n'
                    '              spatial_squeeze=True\n'
                    '              reuse=None,\n'
                    '              scope="resnet_v2_200")\n'
}

name = 'vgg_16'
from models.research.slim.nets.vgg \
    import vgg_16 as net, \
    vgg_arg_scope as arg_scope

NETWORK_LIST[name] = {
    'ckpt_path': networks.CKPT_PATH.format(name),
    'num_classes': 1000,
    'input_size': (224, 224, 3),
    'net': net,
    'arg_scope': arg_scope,
    'kwargs': {},
    'description': 'vgg_16(inputs,\n'
                   '       num_classes=1000,\n'
                   '       is_training=True,\n'
                   '       dropout_keep_prob=0.5,\n'
                   '       spatial_squeeze=True,\n'
                   '       scope="vgg_16"\n'
                   '       fc_conv_padding="VALID",\n'
                   '       global_pool=False)\n'
}

name = 'vgg_19'
from models.research.slim.nets.vgg \
    import vgg_19 as net, \
    vgg_arg_scope as arg_scope

NETWORK_LIST[name] = {
    'ckpt_path': networks.CKPT_PATH.format(name),
    'num_classes': 1000,
    'input_size': (224, 224, 3),
    'net': net,
    'arg_scope': arg_scope,
    'kwargs': {},
    'description': 'vgg_19(inputs,\n'
                   '       num_classes=1000,\n'
                   '       is_training=True,\n'
                   '       dropout_keep_prob=0.5,\n'
                   '       spatial_squeeze=True,\n'
                   '       scope="vgg_19"\n'
                   '       fc_conv_padding="VALID",\n'
                   '       global_pool=False)\n'
}

name = 'nasnet_mobile'
from models.research.slim.nets.nasnet.nasnet \
    import build_nasnet_mobile as net, \
    nasnet_mobile_arg_scope as arg_scope

NETWORK_LIST[name] = {
    'ckpt_path': networks.CKPT_PATH.format('nasnet-a_mobile'),
    'num_classes': 1001,
    'input_size': (224, 224, 3),
    'net': net,
    'arg_scope': arg_scope,
    'kwargs': {},
    'description': 'build_nasnet_mobile(inputs,\n'
                   '                    num_classes=1000,\n'
                   '                    is_training=True,\n'
                   '                    final_endpoint=None,\n'
                   '                    config=None,\n'
                   '                    current_step=None)\n'
}

name = 'nasnet_large'
from models.research.slim.nets.nasnet.nasnet \
    import build_nasnet_large as net, \
    nasnet_large_arg_scope as arg_scope

NETWORK_LIST[name] = {
    'ckpt_path': networks.CKPT_PATH.format(name),
    'num_classes': 1001,
    'input_size': (224, 224, 3),
    'net': net,
    'arg_scope': arg_scope,
    'kwargs': {},
    'description': 'build_nasnet_large(inputs,\n'
                   '                   num_classes=1000,\n'
                   '                   is_training=True,\n'
                   '                   final_endpoint=None,\n'
                   '                   config=None,\n'
                   '                   current_step=None)\n'
}

name = 'pnasnet_mobile'
from models.research.slim.nets.nasnet.pnasnet \
    import build_pnasnet_mobile as net, \
    pnasnet_mobile_arg_scope as arg_scope

NETWORK_LIST[name] = {
    'ckpt_path': networks.CKPT_PATH.format(name),
    'num_classes': 1001,
    'input_size': (224, 224, 3),
    'net': net,
    'arg_scope': arg_scope,
    'kwargs': {},
    'description': 'build_pnasnet_mobile(inputs,\n'
                   '                     num_classes=1000,\n'
                   '                     is_training=True,\n'
                   '                     final_endpoint=None,\n'
                   '                     config=None)\n'
}

name = 'pnasnet_large'
from models.research.slim.nets.nasnet.pnasnet \
    import build_pnasnet_large as net, \
    pnasnet_large_arg_scope as arg_scope

NETWORK_LIST[name] = {
    'ckpt_path': networks.CKPT_PATH.format(name),
    'num_classes': 1001,
    'input_size': (224, 224, 3),
    'net': net,
    'arg_scope': arg_scope,
    'kwargs': {},
    'description': 'build_pnasnet_large(inputs,\n'
                   '                    num_classes=1000,\n'
                   '                    is_training=True,\n'
                   '                    final_endpoint=None,\n'
                   '                    config=None)\n'
}

# need to add mobile net
