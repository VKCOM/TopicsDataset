import numpy as np

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
import pickle

from scipy.stats import entropy

from experiments.datasets.mnist_ds import get_categorical_mnist
from experiments.models.mnist_models import get_qbc_model

(x, y), (x_test, y_test) = get_categorical_mnist()

POOL_SIZE = 10000
INIT_SIZE = 500

n_labeled_examples = x.shape[0]

# RETRAIN_EPOCHS = 20
# BATCH_SIZE = 100

# TEST VALUES
# POOL_SIZE = 200
BATCH_SIZE = 100
INIT_EPOCHS = 50
RETRAIN_EPOCHS = 20


def get_margin(predictions):
    part = np.partition(-predictions, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]
    return margin


def get_entropy(predictions):
    return np.transpose(entropy(np.transpose(predictions)))


for i in range(1, 2):
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    model = get_qbc_model(mc=False)

    x_train = x[training_indices]
    y_train = y[training_indices]

    model.fit(
        x_train, y_train,
        epochs=INIT_EPOCHS,
        validation_data=(x_test, y_test),
        verbose=1
    )

    x_pool = np.delete(x, training_indices, axis=0)
    y_pool = np.delete(y, training_indices, axis=0)

    # next two lines added in second version
    x_pool = x_pool[:POOL_SIZE]
    y_pool = y_pool[:POOL_SIZE]

    entropies = get_entropy(model.predict(x_pool))
    indices_sorted_by_entropy = entropies.argsort()

    x_pool = x_pool[indices_sorted_by_entropy]
    y_pool = y_pool[indices_sorted_by_entropy]
    entropies = entropies[indices_sorted_by_entropy]

    validation_accuracies_after_learning = []

    prediction = model.predict(x_pool)
    losses = keras.losses.categorical_crossentropy(y_pool, prediction)

    for j in range(POOL_SIZE // BATCH_SIZE):
        print('j=', j)
        x_train_current = np.concatenate((x_train, x_pool[j: j + BATCH_SIZE]), axis=0)
        y_train_current = np.concatenate((y_train, y_pool[j: j + BATCH_SIZE]), axis=0)

        print('x_train_current.shape', x_train_current.shape)
        print('y_train_current.shape', y_train_current.shape)

        model_copy = tf.keras.models.clone_model(model)
        model_copy.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer='adam',
            metrics=['accuracy']
        )

        retrain_val_accuracies = []
        for k in range(RETRAIN_EPOCHS):
            model_copy.fit(x_train_current, y_train_current,
                epochs=1,
                verbose=1
            )
            retrain_val_accuracies.append(model_copy.evaluate(x_test, y_test)[1])

        mean_val_accuracy = sum(retrain_val_accuracies) / len(retrain_val_accuracies)
        validation_accuracies_after_learning.append(mean_val_accuracy)
        print('retrain val_accs:', retrain_val_accuracies, '\nmean val acc:', mean_val_accuracy)

    Path('statistic/mnist/ideal_active_learning/').mkdir(parents=True, exist_ok=True)
    pickle.dump(
        {
            'val_accuracies': validation_accuracies_after_learning,
            'entropies': entropies,
            'losses': losses,
            'batch_size': BATCH_SIZE,
            'pool_size': POOL_SIZE
        },
        open('statistic/mnist/ideal_active_learning/big_batches_2.pickle', 'wb')
    )

    print('result:', pickle.load(open('statistic/mnist/ideal_active_learning/big_batches_2.pickle', 'rb')))
