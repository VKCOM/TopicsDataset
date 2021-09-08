from functools import partial, update_wrapper

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.models.topics_torch_models import NormModel, Autoencoder, NormModelBN
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment
from experiments.models.torch_topics_decorator import TopicsDecorator

from modAL import KerasActiveLearner

import torch.optim as optim

from modAL.density import sud, sud_margin_logit

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

INIT_EPOCHS = 45
AUTOENCODER_EPOCHS = 100
N_QUERIES = 200

# TEST VALUES
# INIT_EPOCHS = 1
# AUTOENCODER_EPOCHS = 1
# N_QUERIES = 1

AD = 64  # dimension of autoencoder

sud_params = [1000]

x_img_train_t = torch.tensor(x_img_train).float()
x_txt_train_t = torch.tensor(x_txt_train).float()
x_img_val_t = torch.tensor(x_img_val).float()
x_txt_val_t = torch.tensor(x_txt_val).float()

train_ds = TensorDataset(x_img_train_t, x_txt_train_t)
val_ds = TensorDataset(x_img_val_t, x_txt_val_t)

train_loader = DataLoader(train_ds, batch_size=512)
val_loader = DataLoader(val_ds, batch_size=512)

autoencoder = Autoencoder(d=AD)
autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

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


for i in range(1, 2):
    for sud_param in sud_params:
        np.random.seed(i)
        init_training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

        model = NormModelBN(drop=0.5, d=128)
        model_optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)

        decorated_model = TopicsDecorator(model, model_optimizer)

        x_init_train = [x_img_train[init_training_indices], x_txt_train[init_training_indices]]
        y_init_train = y_train[init_training_indices]

        x_pool = [
            np.delete(x_img_train, init_training_indices, axis=0),
            np.delete(x_txt_train, init_training_indices, axis=0)
        ]
        y_pool = np.delete(y_train, init_training_indices, axis=0)
        original_indices_pool = np.delete(original_indices_train, init_training_indices, axis=0)

        preset_density_transform = update_wrapper(partial(
            sud_margin_logit,
            transform=(
                lambda x: encoder(
                    torch.tensor(x[0]).float(), torch.tensor(x[1]).float()
                ).detach().numpy()
            ),
            n_instances=BATCH_SIZE,
            top_uncertain=sud_param,
            sparse=True,
            uncertainty_density_combination='sum'
        ), sud_margin_logit)

        # now here is KerasActiveLearner because it is suitable also for decorated pytorch models
        learner = KerasActiveLearner(
            estimator=decorated_model,
            X_training=x_init_train,
            y_training=y_init_train,
            query_strategy=preset_density_transform,
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
            name='torch_topics_sud_top' + str(sud_param) + '_trivial_encode_ad64_margin_logit_sparse_sum_i2000_b20_q200_' + str(i)
        )

        experiment.run()
        experiment.save_state(
            'statistic/topics/torch_bn/sud_top' + str(sud_param) + '_trivial_encode_ad64_margin_logit_sparse_sum_i2000_b20_q200_' + str(i))
