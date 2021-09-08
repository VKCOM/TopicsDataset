import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, Multiply, Add, Lambda, Reshape, Flatten, dot
from tensorflow.keras import regularizers

from keras_radam.training import RAdamOptimizer

def create_model_residual_concat():
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x_img = Dense(128, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)

    x_txt = Dense(128, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)

    x = concatenate([x_img, x_txt])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = concatenate([x, x_img, x_txt])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)

    out = Dense(50, activation='softmax')(x)
    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    return model


def get_model_residual_concat():
    model = create_model_residual_concat()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_model_residual_concat_radam():
    model = create_model_residual_concat()
    optimizer = RAdamOptimizer(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model