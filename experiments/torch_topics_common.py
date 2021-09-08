from copy import deepcopy
from functools import partial, update_wrapper

import numpy as np

from modAL.cluster import cluster_sampling, cluster_margin_sampling
from modAL.density import sud
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling, entropy_top_sampling, \
    classifier_margin

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from experiments.models.topics_torch_models import UAModel2, NormModel
from experiments.datasets.topics_ds import get_unpacked_data
from experiments.al_experiment import Experiment

from modAL.passive import passive_strategy
from modAL import KerasActiveLearner

import torch.optim as optim

from modAL.utils import multi_argmax
from modAL.utils.selection import shuffled_argmax
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

# preset_passive = update_wrapper(partial(passive_strategy, n_instances=BATCH_SIZE), passive_strategy)
# preset_least_confident = update_wrapper(partial(uncertainty_sampling, n_instances=BATCH_SIZE), uncertainty_sampling)


def classifier_modified_margin(classifier, X, proba=True, **predict_kwargs) -> np.ndarray:
    return 1 - classifier_margin(classifier, X, proba, **predict_kwargs)


def custom_margin_sampling(classifier, X,
                    n_instances: int = 1, random_tie_break: bool = False,
                    **uncertainty_measure_kwargs):

    margin = classifier_modified_margin(classifier, X, **uncertainty_measure_kwargs)

    query_idx = multi_argmax(margin, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        return query_idx, [x[query_idx] for x in X]
    return query_idx, X[query_idx]


# preset_margin = update_wrapper(partial(margin_sampling, n_instances=BATCH_SIZE), margin_sampling)
# preset_entropy = update_wrapper(partial(entropy_sampling, n_instances=BATCH_SIZE), entropy_sampling)
preset_custom_margin = update_wrapper(partial(custom_margin_sampling, n_instances=BATCH_SIZE), custom_margin_sampling)
# preset_cluster_margin_img = partial(cluster_margin_sampling, transform=(lambda x: x[0]), n_instances=BATCH_SIZE)
# preset_txt = partial(cluster_sampling, transform=(lambda x: x[1]), n_instances=BATCH_SIZE)
# preset_concat = partial(cluster_sampling, transform=(lambda x: np.concatenate(x, axis=1)), n_instances=BATCH_SIZE)
# preset_entropy_top_3 = partial(entropy_top_sampling, n_instances=BATCH_SIZE, n_top=3)
# preset_entropy_top_4 = partial(entropy_top_sampling, n_instances=BATCH_SIZE, n_top=4)
# preset_entropy_top_5 = partial(entropy_top_sampling, n_instances=BATCH_SIZE, n_top=5)

# preset_density_concat_entropy = update_wrapper(partial(sud, transform=(lambda x: np.concatenate(x, axis=1)), n_instances=BATCH_SIZE), sud)
# preset_bald_trident_5 = update_wrapper(partial(bald_trident_based_sampling, cmt_size=5, n_instances=BATCH_SIZE), bald_trident_based_sampling)
# preset_bald_trident_10 = update_wrapper(partial(bald_trident_based_sampling, cmt_size=10, n_instances=BATCH_SIZE), bald_trident_based_sampling)

query_dict = {
    # 'passive': preset_passive,
    # 'density_concat_entropy': preset_density_concat_entropy
    # 'least_confident': preset_least_confident,
    # 'margin': preset_margin,
    'custom_margin': preset_custom_margin
    # 'cluster_margin_img': preset_cluster_margin_img
    # 'entropy': preset_entropy
    # 'cluster_by_img': preset_img,
    # 'entropy_top_3': preset_entropy_top_3,
    # 'entropy_top_4': preset_entropy_top_4,
    # 'entropy_top_5': preset_entropy_top_5
}

for i in range(1, 6):
    np.random.seed(i)
    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=INIT_SIZE)

    x_init_train = [x_img_train[training_indices], x_txt_train[training_indices]]
    y_init_train = y_train[training_indices]

    general_model = NormModel(drop=0.5, d=128)
    general_optimizer = optim.Adam(general_model.parameters(), lr=1e-3, weight_decay=0.0005)

    general_decorated_model = TopicsDecorator(general_model, general_optimizer)
    general_decorated_model.fit(
        X=x_init_train,
        y=y_init_train,
        epochs=INIT_EPOCHS,
        validation_data=([x_img_val, x_txt_val], y_val)
    )

    x_pool = [np.delete(x_img_train, training_indices, axis=0), np.delete(x_txt_train, training_indices, axis=0)]
    y_pool = np.delete(y_train, training_indices, axis=0)

    for query_name in query_dict:
        decorated_model = deepcopy(general_decorated_model)

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
            name='torch_topics_d128_' + query_name + '_i2000_b20_q100_sf512_test_' + str(i),
            bootstrap=False,
            epochs=1
        )

        experiment.run()
        experiment.save_state('statistic/topics/torch/d128/' + query_name + '_i2000_b20_q100_sf512_test_' + str(i))
