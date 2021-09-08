from importlib import resources as pkg_resources
from pickle import UnpicklingError
import json
import pandas as pd
import numpy as np
import pickle
from struct import unpack
from base64 import b64decode
from functools import partial
from keras.utils.np_utils import to_categorical

from experiments import config

IMG_LEN = 1024
TXT_LEN = 300
N_CLASSES = 50

ds_config = json.load(pkg_resources.open_binary(config, 'config.json'))
ds_path = ds_config['ds_path']
unpacked_ds_path = ds_config['unpacked_ds_path']

def get_unpacked_data():
    try:
        topics_pickle_in = open(unpacked_ds_path, 'rb')
        x_img, x_txt, y = pickle.load(topics_pickle_in)
    except (FileNotFoundError, UnpicklingError):
        ds = np.load(ds_path)
        x_txt = ds[:, 1: 1 + TXT_LEN]
        x_img = ds[:, 1 + TXT_LEN:]
        y = to_categorical(ds[:, 0].astype(np.int32))

        topics_pickle_out = open(unpacked_ds_path, "wb")
        pickle.dump((x_img, x_txt, y), topics_pickle_out)
        topics_pickle_out.close()

    return x_img, x_txt, y
