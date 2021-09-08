from copy import deepcopy
from functools import partial, update_wrapper

import numpy as np

import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models.topics_torch_models import NormModel, NormModelTrident
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment
from experiments.models.torch_topics_decorator import TopicsDecorator, TridentDecorator

from modAL import KerasActiveLearner
from modAL.qbc_dropout import qbc_uncertainty_sampling, bald_sampling, bald_modal_sampling, bald_trident_sampling

x_img, x_txt, y = get_unpacked_data()

# print('data loaded')

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

# print('data splited')

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

# print('data scaled')

n_labeled_examples = x_img_train.shape[0]

POOL_SIZE = 100000
INIT_SIZE = 2000
BATCH_SIZE = 20
N_QUERIES = 100
INIT_EPOCHS = 45
CMT_SIZE = 5

# preset_qbc_lc = partial(qbc_uncertainty_sampling, uncertainty_measure='least_confident', cmt_size=CMT_SIZE, n_instances=BATCH_SIZE)
# preset_qbc_margin = partial(qbc_uncertainty_sampling, uncertainty_measure='margin', cmt_size=CMT_SIZE, n_instances=BATCH_SIZE)
# preset_qbc_entropy = partial(qbc_uncertainty_sampling, uncertainty_measure='entropy', cmt_size=CMT_SIZE, n_instances=BATCH_SIZE)
#
# preset_bald_lc = partial(bald_sampling, uncertainty_measure='least_confident', cmt_size=CMT_SIZE, n_instances=BATCH_SIZE)
# preset_bald_margin = partial(bald_sampling, uncertainty_measure='margin', cmt_size=CMT_SIZE, n_instances=BATCH_SIZE)
# preset_bald_entropy = partial(bald_sampling, uncertainty_measure='entropy', cmt_size=CMT_SIZE, n_instances=BATCH_SIZE)

# preset_bald_modal = update_wrapper(partial(bald_modal_sampling, n_instances=BATCH_SIZE), bald_modal_sampling)
# preset_bald_modal_non_mc = update_wrapper(partial(bald_modal_sampling, n_instances=BATCH_SIZE, with_dropout=False), bald_modal_sampling)
# preset_bald_trident_mc = update_wrapper(partial(bald_trident_sampling, n_instances=BATCH_SIZE, with_dropout=True), bald_trident_sampling)
preset_bald_trident_x3 = update_wrapper(partial(bald_trident_sampling, n_instances=BATCH_SIZE, with_dropout=True, cmt_size=3), bald_trident_sampling)

query_dict = {
    'bald_trident_x3': preset_bald_trident_x3
}

# print('queries created')

for i in range(1, 6):
    print('i=', i)
    np.random.seed(i)
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    x_init_train = [x_img_train[training_indices], x_txt_train[training_indices]]
    y_init_train = y_train[training_indices]

    # general_model = NormModel(drop=0.5, d=128)
    # general_optimizer = optim.Adam(general_model.parameters(), lr=1e-3, weight_decay=0.0005)

    trident_model = NormModelTrident(drop=0.5, d=128)
    trident_optimizer = optim.Adam(trident_model.parameters(), lr=1e-3, weight_decay=0.0005)

    # general_decorated_model = TopicsDecorator(general_model, general_optimizer)
    # general_decorated_model.fit(
    #     X=x_init_train,
    #     y=y_init_train,
    #     epochs=INIT_EPOCHS,
    #     validation_data=([x_img_val, x_txt_val], y_val)
    # )

    trident_decorated_model = TridentDecorator(trident_model, trident_optimizer)
    trident_decorated_model.fit(
        X=x_init_train,
        y=y_init_train,
        epochs=INIT_EPOCHS,
        validation_data=([x_img_val, x_txt_val], y_val)
    )

    x_pool = [np.delete(x_img_train, training_indices, axis=0), np.delete(x_txt_train, training_indices, axis=0)]
    y_pool = np.delete(y_train, training_indices, axis=0)

    for query_name in query_dict:
        print('query name =', query_name)
        # if query_name == 'bald_trident':
        decorated_model = deepcopy(trident_decorated_model)
        # else:
        #     decorated_model = deepcopy(general_decorated_model)

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
            name='torch_topics_d128_' + query_name + '_i2000_b20_q100_sf512_' + str(i),
            bootstrap=False,
            epochs=1
        )

        experiment.run()
        experiment.save_state('statistic/topics/torch/d128/' + query_name + '_i2000_b20_q100_sf512_' + str(i))
