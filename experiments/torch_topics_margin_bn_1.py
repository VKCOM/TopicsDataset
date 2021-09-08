from functools import partial, update_wrapper

import numpy as np

from modAL.uncertainty import margin_sampling

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models.topics_torch_models import NormModelBN
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment
from experiments.models.torch_topics_decorator import TopicsDecorator

from modAL import KerasActiveLearner

import torch.optim as optim

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

POOL_SIZE = 100000
INIT_SIZE = 2000
BATCH_SIZE = 20
N_QUERIES = 4900
INIT_EPOCHS = 45
# N_QUERIES = 1
# INIT_EPOCHS = 1

preset_margin = update_wrapper(partial(margin_sampling, n_instances=BATCH_SIZE), margin_sampling)


for i in range(1, 2):
    np.random.seed(i)
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    x_init_train = [x_img_train[training_indices], x_txt_train[training_indices]]
    y_init_train = y_train[training_indices]

    model = NormModelBN(drop=0.5, d=128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)

    decorated_model = TopicsDecorator(model, optimizer)
    decorated_model.fit(
        X=x_init_train,
        y=y_init_train,
        epochs=INIT_EPOCHS,
        validation_data=([x_img_val, x_txt_val], y_val),
        verbose=1
    )

    x_pool = [np.delete(x_img_train, training_indices, axis=0), np.delete(x_txt_train, training_indices, axis=0)]
    y_pool = np.delete(y_train, training_indices, axis=0)

    # now here is KerasActiveLearner because maybe it is suitable also for decorated pytorch models
    learner = KerasActiveLearner(
        estimator=decorated_model,
        X_training=x_init_train,
        y_training=y_init_train,
        query_strategy=preset_margin,
        epochs=0
    )

    experiment = Experiment(
        learner=learner,
        X_pool=x_pool.copy(),
        y_pool=y_pool.copy(),
        X_val=[x_img_val, x_txt_val],
        y_val=y_val,
        n_queries=N_QUERIES,
        random_seed=i,
        pool_size=POOL_SIZE,
        name='torch_topics_bn_margin_i2000_b20_q100_' + str(i),
        intermediate_state_saving=True,
        intermediate_state_filename='statistic/topics/torch_bn/margin_inter_i2000_b20_q100_' + str(i),
        intermediate_state_freq=10,
        bootstrap=False,
        epochs=1,
        verbose=1
    )

    experiment.run()
    experiment.save_state('statistic/topics/torch_bn/margin_i2000_b20_q100_' + str(i))
