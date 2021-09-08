import math
from functools import partial, update_wrapper

import torch
import numpy as np
from scipy.stats import entropy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models.topics_torch_models import NormModelBN
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment
from experiments.models.torch_topics_decorator import TopicsDecorator, LearningLossDecorator, prepare_predict_loader
from experiments.models.learning_loss_models import LearningLossModel3

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
# N_QUERIES = 1
# INIT_EPOCHS = 1
# INIT_SIZE = 2560

RETRAIN_EPOCHS = 1

preset_ll = update_wrapper(partial(learning_loss_strategy, n_instances=BATCH_SIZE), learning_loss_strategy)


# def margin_metric(predictions):
#     part = np.partition(-predictions, 1, axis=1)
#     margin = - part[:, 0] + part[:, 1]
#     return torch.tensor(margin.reshape(-1, 1)).float()

def margin_metric(model, x_img, x_txt):
    model.eval()

    dataloader = prepare_predict_loader([x_img, x_txt])

    predictions = torch.tensor([])
    with torch.no_grad():
        for x_img_cur, x_txt_cur in dataloader:
            outputs = model(x_img_cur.float(), x_txt_cur.float())
            predictions = torch.cat((predictions, outputs), 0)
    predictions = predictions.numpy()

    part = np.partition(-predictions, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]
    return torch.tensor(margin.reshape(-1, 1)).float()


def bald_metric(model, x_img, x_txt):
    cmt_size=3
    predictions = []
    model.train()
    dataloader = prepare_predict_loader([x_img, x_txt])

    for _ in range(cmt_size):
        prediction = torch.tensor([])
        with torch.no_grad():
            for x_img_cur, x_txt_cur in dataloader:
                outputs = model(x_img_cur.float(), x_txt_cur.float())
                prediction = torch.cat((prediction, outputs), 0)
        prediction = prediction.detach().numpy()

        predictions.append(prediction)
    uncertainty = np.transpose(entropy(np.transpose(np.mean(predictions, axis=0))))

    dis_func = np.vectorize(lambda x: 0 if x else x * math.log(x))
    disagreement = np.sum(dis_func(predictions), axis=(0, -1)) / cmt_size

    return torch.tensor((uncertainty + disagreement).reshape(-1, 1)).float()


for i in range(1, 2):
    for n_hidden in range(1, 4):
        np.random.seed(i)
        training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

        x_init_train = [x_img_train[training_indices], x_txt_train[training_indices]]
        y_init_train = y_train[training_indices]

        x_pool = [np.delete(x_img_train, training_indices, axis=0), np.delete(x_txt_train, training_indices, axis=0)]
        y_pool = np.delete(y_train, training_indices, axis=0)

        model = NormModelBN(drop=0.5, d=128)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)
        decorated_model = TopicsDecorator(model, optimizer)

        ll_model = LearningLossModel3(model=model, n_hidden=n_hidden, metrics=[margin_metric, bald_metric], d=4)
        ll_optimizer = optim.Adam(ll_model.parameters(), lr=1e-3, weight_decay=0.0005)
        ll_decorated = LearningLossDecorator(
            decorated_model=decorated_model,
            ll_model=ll_model,
            ll_optimizer=ll_optimizer,
            ll_version=3
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
            name='torch_topics_ll3_margin_bald_n_hidden' + str(n_hidden) + '_i2000_b20_q200_' + str(i),
            intermediate_state_saving=True,
            intermediate_state_filename='statistic/topics/torch_bn/ll3_margin_bald_n_hidden' + str(n_hidden) + 'inter_i2000_b20_q200_' + str(i),
            intermediate_state_freq=10,
            epochs=RETRAIN_EPOCHS,
            verbose=1
        )

        experiment.run()
        experiment.save_state('statistic/topics/torch_bn/ll3_margin_bald_n_hidden' + str(n_hidden) + '_i2000_b20_q200_' + str(i))
