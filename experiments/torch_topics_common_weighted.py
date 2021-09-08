from copy import deepcopy
from functools import partial

import numpy as np

from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling, entropy_top_sampling

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models.topics_torch_models import UAModel2, NormModel
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment

from modAL.passive import passive_strategy
from modAL import KerasActiveLearner

import torch.optim as optim

from models.torch_topics_decorator import TopicsDecorator

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
N_QUERIES = 100
INIT_EPOCHS = 45

preset_passive = partial(passive_strategy, n_instances=BATCH_SIZE)
# preset_least_confident = partial(uncertainty_sampling, n_instances=BATCH_SIZE)
# preset_margin = partial(margin_sampling, n_instances=BATCH_SIZE)
# preset_entropy = partial(entropy_sampling, n_instances=BATCH_SIZE)

query_dict = {
    'passive': preset_passive,
    # 'least_confident': preset_least_confident,
    # 'margin': preset_margin,
    # 'entropy': preset_entropy
}

for i in range(1, 6):
    np.random.seed(i)
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    x_init_train = [x_img_train[training_indices], x_txt_train[training_indices]]
    y_init_train = y_train[training_indices]

    general_model = NormModel(drop=0.5, d=128)
    general_optimizer = optim.Adam(general_model.parameters(), lr=1e-3, weight_decay=0.0005)
    # general_optimizer = optim.SGD(general_model.parameters(), lr=0.5)
    # general_optimizer = optim.Adagrad(general_model.parameters(), lr=1e-2)

    general_decorated_model = TopicsDecorator(general_model, general_optimizer)
    general_decorated_model.fit(
        X=x_init_train,
        y=y_init_train,
        epochs=INIT_EPOCHS,
        validation_data=([x_img_val, x_txt_val], y_val)
    )

    x_pool = [np.delete(x_img_train, training_indices, axis=0), np.delete(x_txt_train, training_indices, axis=0)]
    y_pool = np.delete(y_train, training_indices, axis=0)

    for query_name in query_dict:
        decorated_model = deepcopy(general_decorated_model)

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
            X_val=[x_img_val, x_txt_val],
            y_val=y_val,
            n_queries=N_QUERIES,
            random_seed=i,
            pool_size=POOL_SIZE,
            name='torch_topics_d128_' + query_name + '_i2000_b20_q100_sf512_weighted_' + str(i),
            bootstrap=False,
            epochs=1,
            weight_norm=True
        )

        experiment.run()
        experiment.save_state('statistic/topics/torch/d128/' + query_name + '_i2000_b20_q100_sf512_weighted_' + str(i))
