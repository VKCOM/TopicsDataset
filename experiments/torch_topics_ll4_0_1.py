import pickle
from functools import partial, update_wrapper

import torch
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models.topics_torch_models import NormModelBN
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment
from experiments.models.torch_topics_decorator import TopicsDecorator, LearningLossDecorator
from experiments.models.learning_loss_models import LearningLossModel2, LearningLossModel2_1

from modAL import KerasActiveLearner

import torch.optim as optim

from modAL.learning_loss import learning_loss_strategy

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

N_QUERIES = 200
INIT_EPOCHS = 45

# Test values
# N_QUERIES = 2
# INIT_EPOCHS = 1
RETRAIN_EPOCHS = 1

preset_ll = update_wrapper(partial(learning_loss_strategy, n_instances=BATCH_SIZE), learning_loss_strategy)


def margin_metric(predictions):
    part = np.partition(-predictions, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]
    return torch.tensor(margin.reshape(-1, 1)).float()


for i in range(1, 2):
    for n_hidden in range(2, 3):
        np.random.seed(i)
        training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

        x_init_train = [x_img_train[training_indices], x_txt_train[training_indices]]
        y_init_train = y_train[training_indices]

        x_pool = [np.delete(x_img_train, training_indices, axis=0), np.delete(x_txt_train, training_indices, axis=0)]
        y_pool = np.delete(y_train, training_indices, axis=0)

        model = NormModelBN(drop=0.5, d=128)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)
        decorated_model = TopicsDecorator(model, optimizer)

        ll_model = LearningLossModel2_1(n_hidden=n_hidden, metrics=[margin_metric], d=16, use_raw=True)
        ll_optimizer = optim.Adam(ll_model.parameters(), lr=1e-3, weight_decay=0.0005)
        ll_decorated = LearningLossDecorator(
            decorated_model=decorated_model,
            ll_model=ll_model,
            ll_optimizer=ll_optimizer,
            ll_version=4
        )

        # now here is KerasActiveLearner because maybe it is suitable also for decorated pytorch models
        learner = KerasActiveLearner(
            estimator=ll_decorated,
            X_training=x_init_train,
            y_training=y_init_train,
            query_strategy=preset_ll,
            validation_data=([x_img_val, x_txt_val], y_val),
            epochs=INIT_EPOCHS,
            verbose=1
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
            name='torch_topics_ll4.0_margin_n_hidden' + str(n_hidden) + '_i2000_b20_q200_' + str(i),
            intermediate_state_saving=True,
            intermediate_state_filename='statistic/topics/torch_bn/ll4.0_margin_n_hidden' + str(n_hidden) + 'inter_i2000_b20_q200_' + str(i),
            intermediate_state_freq=10,
            epochs=RETRAIN_EPOCHS,
            verbose=1
        )

        experiment.run()
        experiment.save_state('statistic/topics/torch_bn/ll4.0_margin_n_hidden' + str(n_hidden) + '_i2000_b20_q200_' + str(i))
        pickle.dump(ll_decorated.correllation_history, open('statistic/topics/torch_bn/ll4.0_n_hidden' + str(n_hidden) + '_correlation_history.pkl', 'wb'))
