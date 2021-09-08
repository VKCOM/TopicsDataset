from functools import partial

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from experiments.models.topics_torch_models import UAModel2, NormModel, Autoencoder
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment
from modAL.cluster import cluster_sampling

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

POOL_SIZE = 20000
INIT_SIZE = 2000
BATCH_SIZE = 20
N_QUERIES = 30
INIT_EPOCHS = 45



for i in range(1, 6):
    np.random.seed(i)
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    model = NormModel(drop=0.5, d=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    decorated_model = TopicsDecorator(model, optimizer)

    x_init_train = [x_img_train[training_indices], x_txt_train[training_indices]]
    y_init_train = y_train[training_indices]

    autoencoder = Autoencoder(128)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x_img_init_train = torch.tensor(x_init_train[0]).float()
    x_txt_init_train = torch.tensor(x_init_train[1]).float()
    x_init_ds = TensorDataset(x_img_init_train, x_txt_init_train)
    x_init_loader = DataLoader(x_init_ds, batch_size=2048)

    for epoch in range(20):
        autoencoder.train()
        for x_img_cur, x_txt_cur, _ in x_init_loader:
            autoencoder.zero_grad()
            out_img, out_txt = autoencoder(inp_img=x_img_cur, inp_txt=x_txt_cur)
            loss_img = criterion(out_img, x_img_cur)
            loss_txt = criterion(out_txt, x_txt_cur)
            loss = loss_img + loss_txt

            loss.backward()
            optimizer.step()

    encoder = autoencoder.encoder()

    x_pool = [np.delete(x_img_train, training_indices, axis=0), np.delete(x_txt_train, training_indices, axis=0)]
    y_pool = np.delete(y_train, training_indices, axis=0)

    preset_query = partial(cluster_sampling, n_instances=BATCH_SIZE)

    # now here is KerasActiveLearner because maybe it is suitable also for decorated pytorch models
    learner = KerasActiveLearner(
        estimator=decorated_model,
        X_training=x_init_train,
        y_training=y_init_train,
        query_strategy=preset_query,
        validation_data=([x_img_val, x_txt_val], y_val),
        epochs=INIT_EPOCHS
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
        name='torch_topics_passive_i2000_b20_' + str(i)
    )

    experiment.run()
    experiment.save_state('statistic/topics/torch/passive_i2000_b20_' + str(i))
