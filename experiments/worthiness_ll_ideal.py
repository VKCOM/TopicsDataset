from functools import partial, update_wrapper

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.datasets.topics_worthiness_ds import get_unpacked_data
from experiments.al_experiment import Experiment
from modAL.learning_loss import learning_loss_ideal
from modAL import MultitargetActivateLearner

import torch.optim as optim

from models.torch_worthiness_decorator import WorthinessDecorator
from models.worthiness_models import WorthinessNormModelBN

x_img, x_txt, _, y = get_unpacked_data()
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

# Test values
# N_QUERIES = 5
# INIT_EPOCHS = 1

WEIGHT = torch.tensor([1, 300]).float()
preset_query = update_wrapper(partial(learning_loss_ideal, n_instances=BATCH_SIZE), learning_loss_ideal)

for i in range(1, 2):
    np.random.seed(i)
    init_training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    x_init_train = [x_img_train[init_training_indices], x_txt_train[init_training_indices]]
    y_init_train = y_train[init_training_indices]
    print('in init train objects with worthiness == 1:', np.where(np.argmax(y_init_train, axis=1) == 1)[0].shape[0])

    x_pool = [
        np.delete(x_img_train, init_training_indices, axis=0),
        np.delete(x_txt_train, init_training_indices, axis=0)
    ]
    y_pool = np.delete(y_train, init_training_indices, axis=0)
    original_indices_pool = np.delete(original_indices_train, init_training_indices, axis=0)

    model = WorthinessNormModelBN(drop=0.5, d=128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)

    decorated_model = WorthinessDecorator(model, optimizer)

    learner = MultitargetActivateLearner(
        estimator=decorated_model,
        X_training=x_init_train,
        y_training=y_init_train,
        query_strategy=preset_query,
        validation_data=([x_img_val, x_txt_val], y_val),
        epochs=INIT_EPOCHS,
        weight=WEIGHT,
        verbose=1
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
        name='worthiness_ll_ideal_i2000_b20_q200_weight300_' + str(i),
        epochs=1,
        weight=WEIGHT
    )

    experiment.run()
    experiment.save_state('statistic/worthiness/norm/ll_ideal_i2000_b20_q200_weight300_' + str(i))
