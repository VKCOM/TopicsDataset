from typing import Any

import tensorflow as tf
from modAL.utils.data import modALinput

from modAL.uncertainty import entropy_sampling

from modAL import ActiveLearner
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

import experiments.al_experiment as exp


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

mnist_input_shape = (28, 28, 1)

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=mnist_input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class KerasLearner(ActiveLearner):
    def score(self, X: modALinput, y: modALinput, **score_kwargs) -> Any:
        return self.estimator.evaluate(x_test, y_test, verbose=0)[1]


learner = KerasLearner(
    estimator=model,
    X_training=x_train[:500],
    y_training=y_train[:500],
    query_strategy=entropy_sampling
)

mnist_entropy_exp = exp.Experiment(
    learner=learner,
    X_pool=x_train,
    y_pool=y_train,
    X_val=x_test,
    y_val=y_test,
    n_queries= 20,
    random_seed=1,
    pool_size=10000,
    name='mnist_entropy_exp'
)

mnist_entropy_exp.run()

print(mnist_entropy_exp.performance_history)
print(mnist_entropy_exp.time_per_query_history)
print(mnist_entropy_exp.time_per_fit_history)

