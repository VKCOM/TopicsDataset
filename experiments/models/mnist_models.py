from tensorflow.python.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.python.keras import Model
import tensorflow.python.keras as keras
import tensorflow as tf

num_classes = 10
mnist_input_shape = (28, 28, 1)

def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)

def get_qbc_model(mc=False):
    inp = Input(mnist_input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inp)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = get_dropout(x, p=0.25, mc=mc)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = get_dropout(x, p=0.5, mc=mc)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


EPS = 0.001


def get_learning_loss_model(batch_size=10):
    assert batch_size % 2 == 0, 'batch size should be even'

    def learning_loss_fun(y_true, y_predicted):
        @tf.function
        def loss(x):
            t_i, t_j = x[0][0], x[0][1] # pair of true values
            p_i, p_j = x[1][0], x[1][1] # pair of predicted values
            t = -1.0 * tf.math.sign(t_i - t_j) * (p_i - p_j) + EPS
            t = (t + abs(t)) / 2.0
            return t

        batch_true = tf.reshape(y_true[-batch_size:], [-1, 2])
        batch_predicted = tf.reshape(y_predicted[-batch_size:], [-1, 2])
        # indices = tf.random.uniform(
        #     shape=[batch_size],
        #     maxval=tf.shape(y_true)[0],
        #     dtype=tf.int32)
        # batch_true = tf.reshape(tf.gather(y_true, indices), [-1, 2])
        # batch_predicted = tf.reshape(tf.gather(y_predicted, indices), [-1, 2])

        stacked = tf.stack([batch_true, batch_predicted], axis=1)
        res = tf.reduce_sum(tf.map_fn(loss, stacked))
        return res

    inp = Input(mnist_input_shape)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu')(inp)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    y1 = keras.layers.GlobalAveragePooling2D()(x)
    y1 = Flatten()(y1)
    # y1 = Dense(128, activation='relu')(y1)
    y1 = Dense(64, activation='relu')(y1)

    x = Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu')(inp)
    x = Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    y2 = keras.layers.GlobalAveragePooling2D()(x)
    y2 = Flatten()(y2)
    # y2 = Dense(128, activation='relu')(y2)
    y2 = Dense(64, activation='relu')(y2)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)

    # y3 = Dense(128, activation='relu')(x)
    y3 = Dense(64, activation='relu')(x)
    y = keras.layers.concatenate([y1, y2, y3])
    # y = Dense(128, activation='relu')(y)
    y = Dense(64, activation='relu')(y)

    out_target = Dense(num_classes, activation='softmax', name='target_output')(x)
    out_loss = Dense(1, name='loss_output')(y)

    model = Model(inputs=inp, outputs=[out_target, out_loss])
    model.compile(
        optimizer='adam',
        loss={'target_output' : 'categorical_crossentropy', 'loss_output' : learning_loss_fun},
        loss_weights={'target_output': 1., 'loss_output': 2.},
        metrics=['accuracy']
    )

    return model


def get_like_learning_loss_model():

    inp = Input(mnist_input_shape)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu')(inp)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu')(inp)
    x = Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    out_target = Dense(num_classes, activation='softmax', name='target_output')(x)

    model = Model(inputs=inp, outputs=out_target)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
