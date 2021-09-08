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
N_TOPICS = 50
N_WORTHINESSES = 2


def preset_unpack(l, x):
    return unpack('%df' % l, b64decode(x.encode('utf-8')))


unpack_img = partial(preset_unpack, IMG_LEN)
unpack_txt = partial(preset_unpack, TXT_LEN)

ds_config = json.load(pkg_resources.open_binary(config, 'config.json'))
ds_path = ds_config['topics_worthiness_ds_path']
unpacked_ds_path = ds_config['unpacked_topics_worthiness_ds_path']


def get_unpacked_data():
    try:
        ds_pickle_in = open(unpacked_ds_path, 'rb')
        x_img, x_txt, y_topic, y_worthiness = pickle.load(ds_pickle_in)
    except (FileNotFoundError, UnpicklingError):
        df = pd.read_json(open(ds_path, 'rb'), lines=True)

        x_img = np.stack(df['image_embed'].map(unpack_img), axis=0)
        x_txt = np.stack(df['text_embed'].map(unpack_txt), axis=0)

        y_topic = to_categorical(np.array(df['topic']), N_TOPICS)
        y_worthiness = to_categorical(np.array(df['worthiness']), N_WORTHINESSES)

        ds_pickle_out = open(unpacked_ds_path, "wb")
        pickle.dump((x_img, x_txt, y_topic, y_worthiness), ds_pickle_out)
        ds_pickle_out.close()

    return x_img, x_txt, y_topic, y_worthiness
