from functools import partial, update_wrapper

import numpy as np
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models.topics_torch_models import NormModel
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment

from modAL import KerasActiveLearner

import torch.optim as optim

from models.torch_topics_decorator import TopicsDecorator

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

POOL_SIZE = 20000
INIT_SIZE = 2000
BATCH_SIZE = 20
N_QUERIES = 30
INIT_EPOCHS = 45
RETRAIN_EPOCHS = 1

preset_least_confident = update_wrapper(partial(uncertainty_sampling, n_instances=BATCH_SIZE), uncertainty_sampling)
preset_margin = update_wrapper(partial(margin_sampling, n_instances=BATCH_SIZE), margin_sampling)
preset_entropy = update_wrapper(partial(entropy_sampling, n_instances=BATCH_SIZE), entropy_sampling)

query_dict = {
    'least_confident': preset_least_confident,
    'margin': preset_margin,
    'entropy': preset_entropy
}

for i in range(1, 6):
    for query_name in query_dict:
        np.random.seed(i)
        training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

        model = NormModel(drop=0.5, d=64)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

        decorated_model = TopicsDecorator(model, optimizer)

        x_init_train = [x_img_train[training_indices], x_txt_train[training_indices]]
        y_init_train = y_train[training_indices]

        x_pool = [np.delete(x_img_train, training_indices, axis=0), np.delete(x_txt_train, training_indices, axis=0)]
        y_pool = np.delete(y_train, training_indices, axis=0)
        original_indices_pool = np.delete(original_indices_train, training_indices, axis=0)

        # now here is KerasActiveLearner because maybe it is suitable also for decorated pytorch models
        learner = KerasActiveLearner(
            estimator=decorated_model,
            X_training=x_init_train,
            y_training=y_init_train,
            query_strategy=query_dict[query_name],
            validation_data=([x_img_val, x_txt_val], y_val),
            epochs=INIT_EPOCHS
        )

        experiment = Experiment(
            learner=learner,
            X_pool=x_pool,
            y_pool=y_pool,
            original_indices_pool=original_indices_pool,
            X_val=[x_img_val, x_txt_val],
            y_val=y_val,
            n_queries=N_QUERIES,
            random_seed=i,
            pool_size=POOL_SIZE,
            name='torch_topics_' + query_name + '_i2000_b20_' + str(i),
            intermediate_state_saving=True,
            intermediate_state_filename='statistic/topics/torch/' + query_name + '_i2000_b20_' + str(i),
            intermediate_state_freq=10,
            epochs=RETRAIN_EPOCHS,
            verbose=1
        )

        experiment.run()
        experiment.save_state('statistic/topics/torch/' + query_name + '_i2000_b20_' + str(i))
