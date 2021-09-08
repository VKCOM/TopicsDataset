from functools import partial

import numpy as np
from keras.callbacks import EarlyStopping
from experiments.datasets.mnist_ds import get_categorical_mnist
from modAL import KerasActiveLearner
from modAL.cluster import cluster_sampling
from experiments.models.mnist_models import get_qbc_model
import experiments.al_experiment as exp
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
import tensorflow as tf


def get_mnist_encoder():
    print("=== Preparing MNIST encoder ===")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
    x_train_mnist = x_train_mnist.reshape(x_train_mnist.shape[0], 28, 28, 1)
    x_train_mnist = x_train_mnist.astype('float32')
    x_train_mnist /= 255

    encoding_dim = 20
    input_img = Input(shape=(784,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    x_train_mnist_reshaped = x_train_mnist.reshape((len(x_train_mnist), np.prod(x_train_mnist.shape[1:])))
    x_test_mnist_reshaped = x_test_mnist.reshape((len(x_test_mnist), np.prod(x_test_mnist.shape[1:])))
    autoencoder.fit(x_train_mnist_reshaped, x_train_mnist_reshaped,
                    epochs=50,
                    batch_size=256,
                    validation_data=(x_test_mnist_reshaped, x_test_mnist_reshaped), verbose=0)

    return Model(input_img, encoded)


def reshape_f(x):
    return x.reshape((len(x), np.prod(x.shape[1:])))


def reshape_n_encode(reshape_f, encoder, x):
    x = reshape_f(x)
    return encoder.predict(x)

(x, y), (x_test, y_test) = get_categorical_mnist()

POOL_SIZE = 9500
INIT_SIZE = 500
BATCH_SIZE = 20


encoder = get_mnist_encoder()
n_labeled_examples = x.shape[0]

preset_transform = partial(reshape_n_encode, reshape_f, encoder)
print(np.shape(preset_transform(x)))
preset_batch = partial(cluster_sampling, n_instances=BATCH_SIZE, transform=preset_transform, proba=False)
es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=3)

for i in range(6, 7):
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)
    model = get_qbc_model(mc=False)

    x_train = x[training_indices]
    y_train = y[training_indices]
    x_pool = np.delete(x, training_indices, axis=0)
    y_pool = np.delete(y, training_indices, axis=0)

    learner = KerasActiveLearner(
        estimator=model,
        X_training=x_train,
        y_training=y_train,
        query_strategy=preset_batch,
        validation_data=(x_test, y_test),
        epochs=20,
        callbacks=[es]
    )

    mnist_ranked_batch_exp = exp.Experiment(
        learner=learner,
        X_pool=x_pool,
        y_pool=y_pool,
        X_val=x_test,
        y_val=y_test,
        n_queries=30,
        random_seed=i,
        pool_size=POOL_SIZE,
        name='cluster_bs_i500_b20_' + str(i),
        bootstrap=True
    )

    mnist_ranked_batch_exp.run()

    print(mnist_ranked_batch_exp.performance_history)
    print(mnist_ranked_batch_exp.time_per_query_history)
    print(mnist_ranked_batch_exp.time_per_fit_history)
    mnist_ranked_batch_exp.save_state('statistic/cluster_bs_i500_b20_' + str(i))
