from copy import deepcopy
from functools import partial, update_wrapper

import numpy as np

import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models.topics_torch_models import NormModelTrident, NormModelTridentBN, TridentMTL
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment
from experiments.models.torch_topics_decorator import TridentDecorator, TridentMTLDecorator

from modAL import KerasActiveLearner
from modAL.qbc_dropout import bald_trident_sampling, bald_trident_based_sampling, bald_trident_bn

x_img, x_txt, y = get_unpacked_data()
indices = np.arange(x_img.shape[0])

x_img_train, x_img_test, x_txt_train, x_txt_test, y_train, y_test, original_indices_train, original_indices_test = train_test_split(
    x_img,
    x_txt,
    y,
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)

x_img_train, x_img_val, x_txt_train, x_txt_val, y_train, y_val, original_indices_train, original_indices_val = train_test_split(
    x_img_train,
    x_txt_train,
    y_train,
    original_indices_train,
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

CMT_SIZE = 3

RETRAIN_EPOCHS = 1

# test values
# INIT_SIZE = 19980
# N_QUERIES = 2
# INIT_EPOCHS = 1
# RETRAIN_EPOCHS = 1

preset_bald_trident_x3 = update_wrapper(partial(bald_trident_bn, n_instances=BATCH_SIZE, cmt_size=CMT_SIZE), bald_trident_bn)

query_dict = {
    'bald_trident_bn': preset_bald_trident_x3
}

for i in range(1, 2):
    np.random.seed(i)
    init_training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    x_init_train = [x_img_train[init_training_indices], x_txt_train[init_training_indices]]
    y_init_train = y_train[init_training_indices]

    trident_model = TridentMTL()
    trident_optimizer = optim.Adam(trident_model.parameters(), lr=1e-3, weight_decay=0.0005)

    trident_decorated_model = TridentMTLDecorator(trident_model, trident_optimizer)
    trident_decorated_model.fit(
        X=x_init_train,
        y=y_init_train,
        epochs=INIT_EPOCHS,
        validation_data=([x_img_val, x_txt_val], y_val)
    )

    x_pool = [np.delete(x_img_train, init_training_indices, axis=0), np.delete(x_txt_train, init_training_indices, axis=0)]
    y_pool = np.delete(y_train, init_training_indices, axis=0)
    original_indices_pool = np.delete(original_indices_train, init_training_indices, axis=0)

    for query_name in query_dict:
        decorated_model = deepcopy(trident_decorated_model)

        # now here is KerasActiveLearner because maybe it is suitable also for decorated pytorch models
        learner = KerasActiveLearner(
            estimator=decorated_model,
            X_training=x_init_train,
            y_training=y_init_train,
            query_strategy=query_dict[query_name],
            epochs=0
        )

        experiment = Experiment(
            learner=learner,
            X_pool=x_pool.copy(),
            y_pool=y_pool.copy(),
            original_indices_pool=original_indices_pool,
            X_val=[x_img_val, x_txt_val],
            y_val=y_val,
            n_queries=N_QUERIES,
            random_seed=i,
            pool_size=POOL_SIZE,
            name='torch_topics_mtl_endless_init_bn_' + query_name + '_e1_i2000_b20_' + str(i),
            intermediate_state_saving=True,
            intermediate_state_filename='statistic/topics/torch_mtl_init_bn/' + query_name + '_i2000_b20_q200_inter_' + str(i),
            intermediate_state_freq=10,
            bootstrap=False,
            epochs=RETRAIN_EPOCHS,
            use_batch_norm=False
        )

        experiment.run()
        experiment.save_state('statistic/topics/torch_mtl_init_bn/' + query_name + '_i2000_b20_q200_' + str(i))
