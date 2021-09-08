from copy import deepcopy
from functools import partial, update_wrapper

import numpy as np

import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models.topics_torch_models import NormModelTridentBN
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment
from experiments.models.torch_topics_decorator import TridentAsNormDecorator, TridentDecorator

from modAL import KerasActiveLearner
from modAL.uncertainty import margin_trident_sampling

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
RETRAIN_EPOCHS = 1

N_QUERIES = 4900
INIT_EPOCHS = 45

# test values
# N_QUERIES = 1
# INIT_EPOCHS = 1

preset_margin_sum = update_wrapper(partial(margin_trident_sampling, n_instances=BATCH_SIZE), margin_trident_sampling)
preset_margin_sum_of_squares = update_wrapper(
    partial(margin_trident_sampling, n_instances=BATCH_SIZE, combination=(lambda xs: sum([x ** 2 for x in xs]))),
    margin_trident_sampling
)
preset_margin_multiplication = update_wrapper(
    partial(margin_trident_sampling, n_instances=BATCH_SIZE, combination=(lambda xs: np.prod(xs))),
    margin_trident_sampling
)
query_dict = {
    # 'margin_trident_sum': preset_margin_sum,
    # 'margin_trident_sum_of_squares': preset_margin_sum_of_squares,
    'margin_trident_multiplication': preset_margin_multiplication
}

for i in range(1, 2):
    np.random.seed(i)
    init_training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    x_init_train = [x_img_train[init_training_indices], x_txt_train[init_training_indices]]
    y_init_train = y_train[init_training_indices]

    model = NormModelTridentBN(drop=0.5, d=128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)

    decorated_model = TridentDecorator(model, optimizer)
    decorated_model.fit(
        X=x_init_train,
        y=y_init_train,
        epochs=INIT_EPOCHS,
        validation_data=([x_img_val, x_txt_val], y_val)
    )

    x_pool = [np.delete(x_img_train, init_training_indices, axis=0), np.delete(x_txt_train, init_training_indices, axis=0)]
    y_pool = np.delete(y_train, init_training_indices, axis=0)
    original_indices_pool = np.delete(original_indices_train, init_training_indices, axis=0)

    for query_name in query_dict:
        print('query name =', query_name)
        decorated_model_copy = deepcopy(decorated_model)

        # now here is KerasActiveLearner because maybe it is suitable also for decorated pytorch models
        learner = KerasActiveLearner(
            estimator=decorated_model_copy,
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
            name='torch_topics_' + query_name + '_e1_i2000_b20_2_' + str(i),
            intermediate_state_saving=True,
            intermediate_state_filename='statistic/topics/torch/endless_queries_bn/' + query_name + '_e1_i2000_b20_inter_2_' + str(i),
            intermediate_state_freq=10,
            bootstrap=False,
            epochs=RETRAIN_EPOCHS
        )

        experiment.run()
        experiment.save_state(
            'statistic/topics/torch/endless_queries_bn/' + query_name + '_e1_i2000_b20_2_' + str(i))
