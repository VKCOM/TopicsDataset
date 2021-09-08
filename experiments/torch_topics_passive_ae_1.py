from functools import partial, update_wrapper
import copy
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models.topics_autoencoders import Autoencoder, fit_autoencoder
from experiments.models.after_encoder_models import AfterEncoderModel
from experiments.models.torch_topics_decorator import TopicsDecorator

from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment

from modAL.passive import passive_strategy
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
N_QUERIES = 200

# INIT_EPOCHS = 35
INIT_EPOCHS = 100
AE_EPOCHS = 50
RETRAIN_EPOCHS = 15
ES_DIF = 0.01
ES_TOLERANCE = 2

# Test values
# N_QUERIES = 1
# INIT_EPOCHS = 10
# AE_EPOCHS = 1

preset_query = update_wrapper(partial(passive_strategy, n_instances=BATCH_SIZE), passive_strategy)

autoencoder = Autoencoder(d=128)
autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

fit_autoencoder(
    autoencoder=autoencoder,
    optimizer=autoencoder_optimizer,
    epochs=AE_EPOCHS,
    X_train=[x_img_train, x_txt_train],
    X_val=[x_img_val, x_txt_val]
)

for i in range(1, 2):
    np.random.seed(i)
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    x_init_train = [x_img_train[training_indices], x_txt_train[training_indices]]
    y_init_train = y_train[training_indices]

    x_pool = [np.delete(x_img_train, training_indices, axis=0), np.delete(x_txt_train, training_indices, axis=0)]
    y_pool = np.delete(y_train, training_indices, axis=0)

    model = AfterEncoderModel(encoder=copy.deepcopy(autoencoder.encoder), drop=0.5, d=128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)

    decorated_model = TopicsDecorator(model, optimizer)

    # now here is KerasActiveLearner because maybe it is suitable also for decorated pytorch models
    learner = KerasActiveLearner(
        estimator=decorated_model,
        X_training=x_init_train,
        y_training=y_init_train,
        query_strategy=preset_query,
        validation_data=([x_img_val, x_txt_val], y_val),
        epochs=INIT_EPOCHS,
        es_dif=ES_DIF,
        es_tol=ES_TOLERANCE
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
        name='torch_topics_ae_passive_es01_tol2_i2000_b20_q200_' + str(i),
        intermediate_state_saving=True,
        intermediate_state_filename='statistic/topics/torch_ae/passive_ae_es01_tol2_i2000_b20_q100_sf512_' + str(i),
        intermediate_state_freq=10,
        epochs=RETRAIN_EPOCHS,
        es_dif=ES_DIF,
        es_tol=ES_TOLERANCE
    )

    experiment.run()
    experiment.save_state('statistic/topics/torch_ae/passive_ae_es01_tol2_i2000_b20_q200_' + str(i))
