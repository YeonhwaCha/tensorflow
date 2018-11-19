import glob
import os
import tensorflow as tf

from scripts import networks
from scripts.networks.networks_list import NETWORK_LIST

###############################
# GLOBAL VARIABLES
# file_path : scripts/environments/global_env.py
# CKPT_PATH   : the format which is included .ckpt extension

def Network(name):
    network = NETWORK_LIST[name]
    return network
"""
ckpts = glob.glob(networks.CKPT_PATH.format('*'))
for ckpt in ckpts:
    name = os.path.splitext(os.path.basename(ckpt))[0]
    if name not in networks:
        continue
"""

