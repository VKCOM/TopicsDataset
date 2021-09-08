from functools import partial

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping

from experiments.models.topics_models import get_model_residual_concat_radam
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment

from modAL.passive import passive_strategy
from modAL import KerasActiveLearner


x_img, x_txt, y = get_unpacked_data()

x_img_train, x_img_test, x_txt_train, x_txt_test, y_train, y_test = train_test_split(
    x_img,
    x_txt,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

x_img_train, x_img_val, x_txt_train, x_txt_val, y_train, y_val = train_test_split(
    x_img_train,
    x_txt_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

img_sscaler = StandardScaler()
img_sscaler.fit(x_img_train)

x_img_train = img_sscaler.transform(x_img_train)
x_img_val = img_sscaler.transform(x_img_val)
x_img_test = img_sscaler.transform(x_img_test)

txt_sscaler = StandardScaler()
txt_sscaler.fit(x_txt_train)

x_txt_train = txt_sscaler.transform(x_txt_train)
x_txt_val = txt_sscaler.transform(x_txt_val)
x_txt_test = txt_sscaler.transform(x_txt_test)

n_labeled_examples = x_img_train.shape[0]

POOL_SIZE = 20000
INIT_SIZE = 2000
BATCH_SIZE = 20
N_QUERIES = 30
INIT_EPOCHS = 30

preset_query = partial(passive_strategy, n_instances=BATCH_SIZE)
es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=3)

for i in range(1, 6):
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)
    model = get_model_residual_concat_radam()

    x_init_train = [x_img_train[training_indices], x_txt_train[training_indices]]
    y_init_train = y_train[training_indices]

    x_pool = [np.delete(x_img_train, training_indices, axis=0), np.delete(x_txt_train, training_indices, axis=0)]
    y_pool = np.delete(y_train, training_indices, axis=0)

    learner = KerasActiveLearner(
        estimator=model,
        X_training=x_init_train,
        y_training=y_init_train,
        query_strategy=preset_query,
        validation_data=([x_img_val, x_txt_val], y_val),
        epochs=INIT_EPOCHS,
        callbacks=[es]
    )

    experiment = Experiment(
        learner=learner,
        X_pool=x_pool,
        y_pool=y_pool,
        X_val=[x_img_val, x_txt_val],
        y_val=y_val,
        n_queries=N_QUERIES,
        random_seed=i,
        pool_size=POOL_SIZE,
        name='topics_passive_i500_b20_' + str(i)
    )

    experiment.run()
    experiment.save_state('statistic/topics/keras/passive_i2000_b20_' + str(i))


model_residual_concat_radam = get_model_residual_concat_radam()
