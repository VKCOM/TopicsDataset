from functools import partial, update_wrapper

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models.topics_worthiness_models import MultitargetTridentModelBN
from experiments.models.torch_topics_worthiness_decorator import TridentMultitargetDecorator
from experiments.datasets.topics_worthiness_ds import get_unpacked_data
from experiments.al_experiment import Experiment

from modAL.passive import passive_strategy
from modAL import MultitargetActivateLearner

import torch.optim as optim


x_img, x_txt, y_topic, y_worthiness = get_unpacked_data()

x_img_train, x_img_test, x_txt_train, x_txt_test, y_topic_train, y_topic_test, y_worthiness_train, y_worthiness_test = train_test_split(
    x_img,
    x_txt,
    y_topic,
    y_worthiness,
    test_size=0.2,
    random_state=42,
    stratify=y_topic
)

x_img_train, x_img_val, x_txt_train, x_txt_val, y_topic_train, y_topic_val, y_worthiness_train, y_worthiness_val = train_test_split(
    x_img_train,
    x_txt_train,
    y_topic_train,
    y_worthiness_train,
    test_size=0.2,
    random_state=42,
    stratify=y_topic_train
)

# img_sscaler = StandardScaler()
# img_sscaler.fit(x_img_train)
#
# x_img_train = img_sscaler.transform(x_img_train)
# x_img_val = img_sscaler.transform(x_img_val)
# x_img_test = img_sscaler.transform(x_img_test)
#
# txt_sscaler = StandardScaler()
# txt_sscaler.fit(x_txt_train)
#
# x_txt_train = txt_sscaler.transform(x_txt_train)
# x_txt_val = txt_sscaler.transform(x_txt_val)
# x_txt_test = txt_sscaler.transform(x_txt_test)

n_labeled_examples = x_img_train.shape[0]

POOL_SIZE = 100000
INIT_SIZE = 2000
BATCH_SIZE = 20

# N_QUERIES = 200
# INIT_EPOCHS = 45

# Test values
N_QUERIES = 5
INIT_EPOCHS = 1

preset_query = update_wrapper(partial(passive_strategy, n_instances=BATCH_SIZE), passive_strategy)

for i in range(1, 2):
    np.random.seed(i)
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    x_init_train = [x_img_train[training_indices], x_txt_train[training_indices]]
    y_init_train = [y_topic_train[training_indices], y_worthiness_train[training_indices]]

    x_pool = [np.delete(x_img_train, training_indices, axis=0), np.delete(x_txt_train, training_indices, axis=0)]
    y_pool = [np.delete(y_topic_train, training_indices, axis=0), np.delete(y_worthiness_train, training_indices, axis=0)]

    model = MultitargetTridentModelBN(drop=0.5, d=128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)

    decorated_model = TridentMultitargetDecorator(model, optimizer)

    learner = MultitargetActivateLearner(
        estimator=decorated_model,
        X_training=x_init_train,
        y_training=y_init_train,
        query_strategy=preset_query,
        validation_data=([x_img_val, x_txt_val], [y_topic_val, y_worthiness_val]),
        epochs=INIT_EPOCHS
    )

    experiment = Experiment(
        learner=learner,
        X_pool=x_pool,
        y_pool=y_pool,
        X_val=[x_img_val, x_txt_val],
        y_val=[y_topic_val, y_worthiness_val],
        n_queries=N_QUERIES,
        random_seed=i,
        pool_size=POOL_SIZE,
        name='topics_worthiness_passive_i2000_b20_q200_' + str(i),
        epochs=1
    )

    experiment.run()
    experiment.save_state('statistic/topics_worthiness/trident/passive_i2000_b20_q200_' + str(i))
