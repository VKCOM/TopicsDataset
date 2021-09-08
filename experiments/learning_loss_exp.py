from functools import partial
from keras.callbacks import EarlyStopping

import numpy as np

from modAL.learning_loss import learning_loss_strategy

from experiments.datasets import get_categorical_mnist
from examples.models.mnist_models import get_learning_loss_model
from modAL import LearningLossActiveLearner
import experiments.al_experiment as exp


(x, y), (x_test, y_test) = get_categorical_mnist()

POOL_SIZE = 10000
INIT_SIZE = 100
BATCH_SIZE = 100

n_labeled_examples = x.shape[0]
training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)
x_train = x[training_indices]
y_train = y[training_indices]

preset_learning_loss = partial(learning_loss_strategy, n_instances=BATCH_SIZE)
es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=3)


for i in range(2, 3):
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)
    x_train = x[training_indices]
    y_train = y[training_indices]
    x_pool = np.delete(x, training_indices, axis=0)
    y_pool = np.delete(y, training_indices, axis=0)

    model = get_learning_loss_model(batch_size=BATCH_SIZE)
    learning_loss_learner = LearningLossActiveLearner(
        estimator=model,
        X_training=x_train,
        y_training=y_train,
        query_strategy=preset_learning_loss,
        validation_data=(x_test, y_test),
    )

    learning_loss_exp = exp.Experiment(
        learner=learning_loss_learner,
        X_pool=x_pool,
        y_pool=y_pool,
        X_val=x_test,
        y_val=y_test,
        n_queries=15,
        random_seed=i,
        pool_size=POOL_SIZE,
        name='ll_w64_i100_b100_' + str(i)
    )

    learning_loss_exp.run()
    print('learning loss performance:', learning_loss_exp.performance_history)
    print('learning loss queries times:', learning_loss_exp.time_per_query_history)
    print('learning loss fits times:', learning_loss_exp.time_per_fit_history)
    print('target loss history:', learning_loss_exp.learner.loss_history)
    print('loss prediction loss history', learning_loss_exp.learner.learning_loss_history)
    learning_loss_exp.save_state('statistic/ll_w64_i100_b100_' + str(i))
