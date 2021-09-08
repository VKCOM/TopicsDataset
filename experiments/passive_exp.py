from functools import partial
from keras.callbacks import EarlyStopping

from modAL.passive import passive_strategy

from experiments.datasets import get_categorical_mnist
from modAL import KerasActiveLearner
import numpy as np
import experiments.al_experiment as exp

from examples.models.mnist_models import get_qbc_model


(x, y), (x_test, y_test) = get_categorical_mnist()

POOL_SIZE = 9500
INIT_SIZE = 500
BATCH_SIZE = 10

n_labeled_examples = x.shape[0]

preset_passive = partial(passive_strategy, n_instances=BATCH_SIZE)
es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=3)

for i in range(1, 6):
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
        query_strategy=preset_passive,
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
        name='passive_i500_b10_' + str(i)
    )

    mnist_ranked_batch_exp.run()

    print(mnist_ranked_batch_exp.performance_history)
    print(mnist_ranked_batch_exp.time_per_query_history)
    print(mnist_ranked_batch_exp.time_per_fit_history)
    mnist_ranked_batch_exp.save_state('statistic/passive_i500_b10_' + str(i))
