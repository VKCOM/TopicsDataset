from functools import partial

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.models.topics_torch_models import NormModel, Autoencoder
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment

from modAL.cluster import cluster_sampling, cluster_margin_sampling
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
AUTOENCODER_EPOCHS = 100
#
# preset_img = partial(cluster_sampling, transform=(lambda x: x[0]), n_instances=BATCH_SIZE)
# preset_txt = partial(cluster_sampling, transform=(lambda x: x[1]), n_instances=BATCH_SIZE)
# preset_concat = partial(cluster_sampling, transform=(lambda x: np.concatenate(x, axis=1)), n_instances=BATCH_SIZE)
#
# query_dict = {
#     'cluster_by_concat_img_txt': preset_concat,
#     'cluster_by_img': preset_img,
#     'cluster_by_txt': preset_txt
# }

x_img_val_t = torch.tensor(x_img_val).float()
x_txt_val_t = torch.tensor(x_txt_val).float()
val_ds = TensorDataset(x_img_val_t, x_txt_val_t)
val_loader = DataLoader(val_ds, batch_size=512)

for i in range(1, 6):
    np.random.seed(i)
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    model = NormModel(drop=0.5, d=128)
    model_optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)

    decorated_model = TopicsDecorator(model, model_optimizer)
    x_init_train = [x_img_train[training_indices], x_txt_train[training_indices]]
    y_init_train = y_train[training_indices]

    autoencoder = Autoencoder(d=8)
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x_img_train_t = torch.tensor(x_init_train[0]).float()
    x_txt_train_t = torch.tensor(x_init_train[1]).float()

    train_ds = TensorDataset(x_img_train_t, x_txt_train_t)
    train_loader = DataLoader(train_ds, batch_size=512)

    for epoch in range(AUTOENCODER_EPOCHS):
        autoencoder.train()

        loss_sum = 0.0
        loss_count = 0
        for x_img_cur, x_txt_cur in train_loader:
            autoencoder.zero_grad()
            out_img, out_txt = autoencoder(inp_img=x_img_cur, inp_txt=x_txt_cur)
            loss_img = criterion(out_img, x_img_cur)
            loss_txt = criterion(out_txt, x_txt_cur)
            loss = loss_img + loss_txt

            loss_sum += loss
            loss_count += 1

            loss.backward()
            autoencoder_optimizer.step()

        print('autoencoder epoch', epoch, 'avg loss =', loss_sum / loss_count)

        autoencoder.eval()

        val_loss_img_sum = 0.0
        val_loss_txt_sum = 0.0
        val_loss_sum = 0.0
        val_loss_count = 0

        with torch.no_grad():
            for x_img_cur, x_txt_cur in val_loader:
                out_img, out_txt = autoencoder(x_img_cur, x_txt_cur)
                loss_img = criterion(out_img, x_img_cur)
                loss_txt = criterion(out_txt, x_txt_cur)
                loss = loss_img + loss_txt

                val_loss_img_sum += loss_img
                val_loss_txt_sum += loss_txt
                val_loss_sum += loss
                val_loss_count += 1

        print(
            'val img loss:', val_loss_img_sum / loss_count,
            'txt_loss:', val_loss_txt_sum / loss_count,
            'img + txt loss', val_loss_sum / loss_count
        )
    encoder = autoencoder.encoder

    x_pool = [np.delete(x_img_train, training_indices, axis=0), np.delete(x_txt_train, training_indices, axis=0)]
    y_pool = np.delete(y_train, training_indices, axis=0)

    preset_cluster_transform = partial(
        cluster_sampling,
        transform=(
            lambda x: encoder(
                torch.tensor(x[0]).float(), torch.tensor(x[1]).float()
            ).detach().numpy()
        ),
        n_instances=BATCH_SIZE)

    # now here is KerasActiveLearner because it is suitable also for decorated pytorch models
    learner = KerasActiveLearner(
        estimator=decorated_model,
        X_training=x_init_train,
        y_training=y_init_train,
        query_strategy=preset_cluster_transform,
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
        name='torch_topics_cluster_trivial_encode_entropy_i2000_b20_q100' + str(i),
        autoencoder=autoencoder,
        autoencoder_optim=autoencoder_optimizer
    )

    experiment.run()
    experiment.save_state('statistic/topics/torch/d128/cluster_trivial_encode_entropy_i2000_b20_q100_' + str(i))
