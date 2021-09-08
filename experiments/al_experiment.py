from typing import Union
from pathlib import Path

import torch
from sklearn.utils import shuffle

import random
import numpy as np
import scipy.sparse as sp
import pickle
import time
import logging

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from modAL import LearningLossActiveLearner
from modAL.models.base import BaseLearner, BaseCommittee

random_name_length = 5


def get_random_name():
    return ''.join(map(str, np.random.randint(low=0, high=9, size=random_name_length)))


def is_multimodal(X):
    return isinstance(X, list) and isinstance(X[0], np.ndarray)


class Experiment:

    def __init__(
            self,
            learner: Union[BaseLearner, BaseCommittee],
            X_pool: Union[np.ndarray, sp.csr_matrix, list],
            y_pool: Union[np.ndarray, sp.csr_matrix, list],
            original_indices_pool: np.ndarray,
            X_val: Union[np.ndarray, sp.csr_matrix, list] = None,
            y_val: Union[np.ndarray, sp.csr_matrix, list] = None,
            n_instances: int = 1,
            n_queries: int = 10,
            random_seed: int = random.randint(0, 100),
            pool_size: int = -1,
            name: str = get_random_name(),
            autoencoder=None,
            autoencoder_optim=None,
            intermediate_state_saving=False,
            intermediate_state_filename=None,
            intermediate_state_freq=1,
            **teach_kwargs
    ):
        self.learner = learner
        self.n_queries = n_queries
        self.random_seed = random_seed
        self.n_instances = n_instances
        self.autoencoder = autoencoder
        self.autoencoder_optim = autoencoder_optim
        self.intermediate_state_saving = intermediate_state_saving

        if self.intermediate_state_saving and intermediate_state_filename is None:
            raise ValueError("intermediate state can't be saved without intermediate_state_filename argument ")
        self.intermediate_state_filename = intermediate_state_filename

        self.intermediate_state_freq = intermediate_state_freq

        # self.init_size = self.learner.X_training.shape[0]

        np.random.seed(random_seed)
        n_instances = X_pool[0].shape[0] if is_multimodal(X_pool) else X_pool.shape[0]
        idx = np.random.choice(range(n_instances), n_instances, replace=False)
        X_pool = [x[idx] for x in X_pool] if is_multimodal(X_pool) else X_pool[idx]
        y_pool = [y[idx] for y in y_pool] if is_multimodal(y_pool) else y_pool[idx]
        original_indices_pool = original_indices_pool[idx]

        real_size = X_pool[0].shape[0] if is_multimodal(X_pool) else X_pool.shape[0]

        if 0 <= pool_size < real_size:
            if is_multimodal(X_pool):
                X_pool = [x[:pool_size] for x in X_pool]
            else:
                X_pool = X_pool[:pool_size]

            if is_multimodal(y_pool):
                y_pool = [y[:pool_size] for y in y_pool]
            else:
                y_pool = y_pool[:pool_size]

            original_indices_pool = original_indices_pool[:pool_size]
        else:
            pool_size = real_size

        self.pool_size = pool_size
        self.X_pool = X_pool
        self.y_pool = y_pool

        self.original_indices_pool = original_indices_pool

        self.X_val = X_pool if X_val is None else X_val
        self.y_val = y_pool if y_val is None else y_val

        self.performance_history = []
        self.time_per_query_history = []
        self.time_per_fit_history = []

        self.name = name
        self._setup_logger()
        self.teach_kwargs = teach_kwargs

    def _setup_logger(self):
        self.logger = logging.getLogger('exp_' + self.name)
        self.logger.setLevel(logging.INFO)
        Path("log").mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler('log/exp_' + self.name + '.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _out_of_data_warn(self):
        self.logger.warning('pool does not have enough data, batch size = '
                            + str(self.n_instances)
                            + ' but pool size = '
                            + str(self.pool_size))

    def save_state(self, state_name):
        state = {
            # 'init_size' : self.init_size,
            'n_instances': self.n_instances,
            'n_queries': self.n_queries,
            'performance_history': self.performance_history,
            'time_per_query_history': self.time_per_query_history,
            'time_per_fit_history': self.time_per_fit_history
        }
        if isinstance(self.learner, LearningLossActiveLearner):
            state['loss_history'] = self.learner.loss_history
            state['learning_loss_history'] = self.learner.learning_loss_history
        Path('/'.join(state_name.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        with open(state_name + '.pkl', 'wb') as f:
            pickle.dump(state, f)

    def save_current_state(self, cur_step):

        if cur_step % self.intermediate_state_freq != 0:
            return

        state = {
            'n_instances': self.n_instances,
            'cur_n_queries': cur_step + 1,
            'performance_history': self.performance_history,
            'time_per_query_history': self.time_per_query_history,
            'time_per_fit_history': self.time_per_fit_history
        }

        Path('/'.join(self.intermediate_state_filename.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        with open(self.intermediate_state_filename + '.pkl', 'wb') as f:
            pickle.dump(state, f)

        self.logger.info('state on step ' + str(cur_step) + ' saved')


    # NOT FOR ANYTHING EXCEPT TOPICS TASK!
    def fit_autoencoder(self):
        criterion = nn.MSELoss()

        x_img_train_t = torch.tensor(self.X_pool[0]).float()
        x_txt_train_t = torch.tensor(self.X_pool[1]).float()

        train_ds = TensorDataset(x_img_train_t, x_txt_train_t)
        train_loader = DataLoader(train_ds, batch_size=512)

        self.autoencoder.train()

        loss_sum = 0.0
        loss_count = 0
        for x_img_cur, x_txt_cur in train_loader:
            self.autoencoder.zero_grad()
            out_img, out_txt = self.autoencoder(inp_img=x_img_cur, inp_txt=x_txt_cur)
            loss_img = criterion(out_img, x_img_cur)
            loss_txt = criterion(out_txt, x_txt_cur)
            loss = loss_img + loss_txt

            loss_sum += loss
            loss_count += 1

            loss.backward()
            self.autoencoder_optim.step()

        self.logger.info('autoencoder train loss ' + str(loss_sum/loss_count))

    def step(self, i=-1):
        self.logger.info('start step #' + str(i))
        start_time_step = time.time()
        if self.n_instances > self.pool_size:
            self._out_of_data_warn()
            return
        start_time_query = time.time()

        if self.learner.query_strategy.__name__ == 'diversity_sampling':
            query_index, query_instance = self.learner.query(self.X_pool, labeled_pool=self.learner.X_training)
        elif 'learning_loss_ideal' in self.learner.query_strategy.__name__:
            query_index, query_instance = self.learner.query(self.X_pool, self.y_pool)
        else:
            query_index, query_instance = self.learner.query(self.X_pool)

        self.logger.info('query idx: ' + str(query_index))
        self.time_per_query_history.append(time.time() - start_time_query)

        if is_multimodal(self.X_pool):
            X_query = [x[query_index] for x in self.X_pool]
        else:
            X_query = self.X_pool[query_index]

        if is_multimodal(self.y_pool):
            y_query = [y[query_index] for y in self.y_pool]
        else:
            y_query = self.y_pool[query_index]

        original_indices_query = self.original_indices_pool[query_index]
        self.logger.info('query topics: ' + str(np.argmax(y_query if not is_multimodal(self.y_pool) else y_query[0], axis=1)))
        self.logger.info('query original indices: ' + str(original_indices_query))

        start_time_fit = time.time()
        self.learner.teach(X=X_query, y=y_query, **self.teach_kwargs)
        self.time_per_fit_history.append(time.time() - start_time_fit)

        if is_multimodal(self.X_pool):
            self.X_pool = [np.delete(x, query_index, axis=0) for x in self.X_pool]
        else:
            self.X_pool = np.delete(self.X_pool, query_index, axis=0)

        if is_multimodal(self.y_pool):
            self.y_pool = [np.delete(y, query_index, axis=0) for y in self.y_pool]
        else:
            self.y_pool = np.delete(self.y_pool, query_index, axis=0)

        self.original_indices_pool = np.delete(self.original_indices_pool, query_index, axis=0)

        if self.autoencoder is not None:
            self.fit_autoencoder()

        score = self.learner.score(self.X_val, self.y_val)
        self.performance_history.append(score)
        self.logger.info('finish step #' + str(i) + ' for ' + str(time.time() - start_time_step) + ' sec')
        if not isinstance(score, dict):
            self.logger.info('current val_accuracy: ' + str(score))
        else:
            for k, v in score.items():
                self.logger.info('current val ' + k + ': ' + str(v))

        if self.intermediate_state_saving or is_multimodal(self.X_pool) and self.X_pool[0].shape[0] == 0:
            self.save_current_state(cur_step=i)

        return query_index, query_instance, score

    def run(self):
        self.logger.info('start experiment process')
        start_time = time.time()
        score = self.learner.score(self.X_val, self.y_val)
        self.performance_history.append(score)
        self.logger.info('initial val_accuracy: ' + str(score))
        for i in range(self.n_queries):
            if self.n_instances > self.pool_size:
                self._out_of_data_warn()
                break
            self.step(i)
        self.logger.info('finish experiment process for ' + str(time.time() - start_time) + ' sec')
        return self.performance_history
