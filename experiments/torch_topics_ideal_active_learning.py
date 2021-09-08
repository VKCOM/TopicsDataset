import copy
from pathlib import Path

import numpy as np

from experiments.models.topics_torch_models import NormModelBN
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.models.torch_topics_decorator import TopicsDecorator

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch.optim as optim

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

POOL_SIZE = 1000
INIT_SIZE = 2000
INIT_EPOCHS = 45
RETRAIN_EPOCHS = 20

# TEST VALUES
# POOL_SIZE = 2
# INIT_EPOCHS = 1
# RETRAIN_EPOCHS = 1

def get_margin(predictions):
    part = np.partition(-predictions, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]
    return margin


for i in range(1, 2):
    np.random.seed(i)
    init_training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    x_init_train = [x_img_train[init_training_indices], x_txt_train[init_training_indices]]
    y_init_train = y_train[init_training_indices]

    model = NormModelBN(drop=0.5, d=128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)

    decorated_model = TopicsDecorator(model, optimizer)
    decorated_model.fit(
        X=x_init_train,
        y=y_init_train,
        epochs=INIT_EPOCHS,
        validation_data=([x_img_val, x_txt_val], y_val),
        verbose=1
    )

    init_val_accuracy = decorated_model.evaluate([x_img_val, x_txt_val], y_val)[1]
    print('init validation accuracy:', init_val_accuracy)

    x_pool = [
        np.delete(x_img_train, init_training_indices, axis=0),
        np.delete(x_txt_train, init_training_indices, axis=0)
    ]
    y_pool = np.delete(y_train, init_training_indices, axis=0)
    original_indices_pool = np.delete(original_indices_train, init_training_indices, axis=0)

    x_pool = [x[:POOL_SIZE] for x in x_pool]
    y_pool = y_pool[:POOL_SIZE]

    validation_accuracies_after_learning = []
    margins = []

    for j in range(POOL_SIZE):

        x_train_current = [np.concatenate((x, x_p[j].reshape(1, -1)), axis=0) for x, x_p in zip(x_init_train, x_pool)]
        y_train_current = np.concatenate((y_init_train, y_pool[j].reshape(1, -1)), axis=0)
        decorated_model_copy = copy.deepcopy(decorated_model)

        margins.append(
            get_margin(decorated_model_copy.predict_proba([x_p[j].reshape(1, -1) for x_p in x_pool]))[0]
        )

        retrain_val_accuracies = []
        for k in range(RETRAIN_EPOCHS):
            decorated_model_copy.fit(
                X=x_train_current,
                y=y_train_current,
                epochs=1,
                verbose=1
            )
            retrain_val_accuracies.append(decorated_model_copy.evaluate([x_img_val, x_txt_val], y_val)[1])

        mean_val_accuracy = sum(retrain_val_accuracies) / len(retrain_val_accuracies)
        validation_accuracies_after_learning.append(mean_val_accuracy)
        print('retrain val_accs:', retrain_val_accuracies, '\nmean val acc:', mean_val_accuracy)

    Path('statistic/topics/torch_bn/ideal_active_learning/').mkdir(parents=True, exist_ok=True)
    pickle.dump(
        {
            'val_accuracies': validation_accuracies_after_learning,
            'margins': margins
        },
        open('statistic/topics/torch_bn/ideal_active_learning/val_accuracies.pickle', 'wb')
    )
