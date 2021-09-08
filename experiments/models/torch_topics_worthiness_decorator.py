import scipy
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np

IMG_LEN = 1024
TXT_LEN = 300
BATCH_SIZE = 512


def prepare_train_loader(X, y):
    x_img = X[0]
    x_txt = X[1]
    y_topic = y[0]
    y_worthiness = y[1]

    p = np.random.permutation(len(x_img))
    x_img = x_img[p]
    x_txt = x_txt[p]
    y_topic = y_topic[p]
    y_worthiness = y_worthiness[p]

    x_img_train_t = torch.tensor(x_img).float()
    x_txt_train_t = torch.tensor(x_txt).float()
    y_topic_train_t = torch.tensor(y_topic)
    y_worthiness_train_t = torch.tensor(y_worthiness)
    train_ds = TensorDataset(x_img_train_t, x_txt_train_t, y_topic_train_t, y_worthiness_train_t)

    cur_batch_size = BATCH_SIZE + (len(x_img) % BATCH_SIZE) // (len(x_img) // BATCH_SIZE)
    cur_batch_size += 2 if not cur_batch_size % 2 else 1

    train_loader = DataLoader(train_ds, batch_size=cur_batch_size)
    return train_loader


def prepare_val_loader(validation_data):
    x_img_val_t = torch.tensor(validation_data[0][0]).float()
    x_txt_val_t = torch.tensor(validation_data[0][1]).float()
    y_topic_val_t = torch.tensor(validation_data[1][0])
    y_worthiness_val_t = torch.tensor(validation_data[1][1])
    val_ds = TensorDataset(x_img_val_t, x_txt_val_t, y_topic_val_t, y_worthiness_val_t)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    return val_loader


def prepare_predict_loader(X):
    x_img_t = torch.tensor(X[0]).float()
    x_txt_t = torch.tensor(X[1]).float()
    ds = TensorDataset(x_img_t, x_txt_t)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    return loader


def calculate_metrics(y_topic_predicted, y_worthiness_predicted, y_topic_target, y_worthiness_target, verbose=1):
    loss_topic = F.nll_loss(
        torch.from_numpy(y_topic_predicted),
        torch.argmax(torch.from_numpy(y_topic_target), dim=1)
    )
    loss_worthiness = F.nll_loss(
        torch.from_numpy(y_worthiness_predicted),
        torch.argmax(torch.from_numpy(y_worthiness_target), dim=1)
    )

    correct_topic = 0
    correct_worthiness = 0

    total = 0
    y_topic_predicted_non_cat = y_topic_predicted.argmax(axis=1)
    y_topic_target_non_cat = y_topic_target.argmax(axis=1)

    y_worthiness_predicted_non_cat = y_worthiness_predicted.argmax(axis=1)
    y_worthiness_target_non_cat = y_worthiness_target.argmax(axis=1)

    for i in range(y_topic_predicted.shape[0]):
        if y_topic_predicted_non_cat[i] == y_topic_target_non_cat[i]:
            correct_topic += 1
        total += 1

    for i in range(y_worthiness_predicted.shape[0]):
        if y_worthiness_predicted_non_cat[i] == y_worthiness_target_non_cat[i]:
            correct_worthiness += 1

    worthiness_roc_auc_score = roc_auc_score(y_worthiness_target, y_worthiness_predicted)
    precision, recall, threshold = precision_recall_curve(
        np.argmax(y_worthiness_target, axis=1),
        np.argmax(y_worthiness_predicted, axis=1)
    )
    worthiness_pr_auc_score = auc(recall, precision)

    if verbose != 0:
        print('val topic acc', correct_topic / total)
        print('val worthiness acc', correct_worthiness / total)
        print('worthiness roc auc score', worthiness_roc_auc_score)
        print('worthiness pr auc score', worthiness_pr_auc_score)

    return {
        'loss_topic': loss_topic,
        'loss_worthiness': loss_worthiness,
        'accuracy_topic': correct_topic / total,
        'accuracy_worthiness': correct_worthiness / total,
        'roc_auc_worthiness': worthiness_roc_auc_score,
        'pr_auc_worthiness': worthiness_pr_auc_score
    }


class MultitargetDecorator:
    """
    implies that
    X == [x_img, x_txt], where x_img and x_txt are numpy arrays
    y == [y_topic, y_worthiness], where y_topic and y_worthiness are numpy arrays
    """

    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    # implied that early stopping is only on train loss
    def fit(self, X, y, epochs=1, validation_data=None, es_dif=None, es_tol=0, verbose=0, topic_coef=1):

        if verbose != 0:
            print('fit on ' + str(X[0].shape[0]) + ' objects')

        train_loader = prepare_train_loader(X, y)

        if validation_data is not None:
            val_loader = prepare_val_loader(validation_data)

        prev_train_loss = None
        tol_epochs = 0

        for epoch in range(epochs):
            self.model.train()

            train_loss_sum = 0.0
            train_loss_count = 0

            for x_img_cur, x_txt_cur, y_topic_cur, y_worthiness_cur in train_loader:

                self.optimizer.zero_grad()
                output_topic, output_worthiness = self.model(x_img_cur, x_txt_cur)

                train_loss_topic = F.nll_loss(output_topic, torch.argmax(y_topic_cur, dim=1))
                train_loss_worthiness = F.nll_loss(output_worthiness, torch.argmax(y_worthiness_cur, dim=1))

                train_loss = topic_coef * train_loss_topic + train_loss_worthiness
                train_loss.backward()

                train_loss_sum += train_loss
                train_loss_count += 1

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

            avg_train_loss = train_loss_sum / train_loss_count
            if verbose != 0:
                print('epoch:', epoch, 'train_loss:', train_loss, 'average train loss', avg_train_loss)

            if validation_data is not None:
                self.model.eval()

                correct_topic = 0
                correct_worthiness = 0
                total = 0
                val_loss_sum = 0.0
                val_loss_count = 0
                worthiness_predictions = torch.tensor([])

                with torch.no_grad():
                    for x_img_cur, x_txt_cur, y_topic_cur, y_worthiness_cur in val_loader:
                        output_topic, output_worthiness = self.model(x_img_cur, x_txt_cur)
                        worthiness_predictions = torch.cat((worthiness_predictions, output_worthiness), 0)
                        val_loss_topic = F.nll_loss(output_topic, torch.argmax(y_topic_cur, dim=1))
                        val_loss_worthiness = F.nll_loss(output_topic, torch.argmax(y_worthiness_cur, dim=1))
                        val_loss = topic_coef * val_loss_topic + val_loss_worthiness

                        val_loss_sum += val_loss
                        val_loss_count += 1
                        for idx, i in enumerate(output_topic):
                            if torch.argmax(i) == torch.argmax(y_topic_cur, dim=1)[idx]:
                                correct_topic += 1
                            total += 1
                        for idx, i in enumerate(output_worthiness):
                            if torch.argmax(i) == torch.argmax(y_worthiness_cur, dim=1)[idx]:
                                correct_worthiness += 1

                worthiness_predictions = worthiness_predictions.numpy()
                if verbose != 0:
                    print('val_acc_topic:', correct_topic / total,
                          'val_acc_worthiness:', correct_worthiness / total,
                          'val_avg_loss:', val_loss_sum / val_loss_count,
                          'roc auc worthiness:', roc_auc_score(validation_data[1][1], worthiness_predictions))

            # es part
            if es_dif is not None and epoch != 0:
                if tol_epochs != 0:  # already in tolerance mode
                    if prev_train_loss - avg_train_loss > es_dif:  # leave tolerance mode
                        tol_epochs = 0
                    elif tol_epochs >= es_tol:  # tolerance limit exceeded
                        return
                    else:  # continue tolerance mode
                       tol_epochs += 1
                elif prev_train_loss - avg_train_loss <= es_dif:  # not in tolerance but to slow learning
                    if es_tol == 0:  # no tolerance
                        return
                    else:  # enter tolerance mode
                        tol_epochs += 1
            prev_train_loss = avg_train_loss

    def predict(self, X, with_dropout=False):

        if with_dropout:
            self.model.train()
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
        else:
            self.model.eval()

        dataloader = prepare_predict_loader(X)

        predictions_topic = torch.tensor([])
        predictions_worthiness = torch.tensor([])

        with torch.no_grad():
            for x_img_cur, x_txt_cur in dataloader:
                output_topic, output_worthiness = self.model(x_img_cur.float(), x_txt_cur.float())
                predictions_topic = torch.cat((predictions_topic, output_topic), 0)
                predictions_worthiness = torch.cat((predictions_worthiness, output_worthiness), 0)

        return predictions_topic.numpy(), predictions_worthiness.numpy()

    def predict_proba(self, X, **predict_kwargs):
        y_topic_predicted, y_worthiness_predicted = self.predict(X, **predict_kwargs)
        return np.exp(y_topic_predicted), np.exp(y_worthiness_predicted)

    def evaluate(self, X, y, verbose=1):
        y_topic_predicted, y_worthiness_predicted = self.predict(X)
        y_topic_target, y_worthiness_target = y
        return calculate_metrics(
            y_topic_predicted=y_topic_predicted,
            y_worthiness_predicted=y_worthiness_predicted,
            y_topic_target=y_topic_target,
            y_worthiness_target=y_worthiness_target
        )


class TridentMultitargetDecorator:
    """
    implies that X == [x_img, x_txt], where x_img and x_txt are numpy arrays
    """

    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, X, y, epochs=1, validation_data=None, es_dif=None, es_tol=0, verbose=1, topic_coef=1, weight=None, use_batch_norm=True):
        if verbose != 0:
            print('fit on', X[0].shape[0], 'objects')
        train_loader = prepare_train_loader(X, y)

        if validation_data is not None:
            val_loader = prepare_val_loader(validation_data)

        prev_train_loss = None
        tol_epochs = 0
        for epoch in range(epochs):
            self.model.train()
            if not use_batch_norm:
                for module in self.model.modules():
                    if isinstance(module, torch.nn.modules.BatchNorm1d):
                        module.eval()
            loss_sum = 0.0
            loss_count = 0

            for x_img_cur, x_txt_cur, y_topic_cur, y_worthiness_cur in train_loader:

                self.optimizer.zero_grad()
                out_common, out_img, out_txt, out_worthiness = self.model(x_img_cur, x_txt_cur)
                topic_target = torch.argmax(y_topic_cur, dim=1)
                worthiness_target = torch.argmax(y_worthiness_cur, dim=1)

                loss_common = F.nll_loss(out_common, topic_target)
                loss_img = F.nll_loss(out_img, topic_target)
                loss_txt = F.nll_loss(out_txt, topic_target)
                loss_topic = (loss_common + loss_img + loss_txt) / 3.0

                loss_worthiness = F.nll_loss(out_worthiness, worthiness_target, weight=weight)
                loss = topic_coef * loss_topic + loss_worthiness
                loss.backward()
                loss_sum += loss

                loss_count += 1

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

            avg_train_loss = loss_sum / loss_count
            print('epoch:', epoch, 'train_loss:', loss, 'average train loss', loss_sum / loss_count)

            if validation_data is not None:
                self.model.eval()

                correct_topic = 0
                correct_worthiness = 0
                total = 0
                loss_sum = 0.0
                loss_count = 0

                with torch.no_grad():
                    for x_img_cur, x_txt_cur, y_topic_cur, y_worthiness_cur in val_loader:
                        out_common, _, _, out_worthiness = self.model(x_img_cur, x_txt_cur)
                        loss_topic = F.nll_loss(out_common, torch.argmax(y_topic_cur, dim=1))
                        loss_worthiness = F.nll_loss(out_worthiness, torch.argmax(y_worthiness_cur, dim=1), weight=weight)
                        loss = topic_coef * loss_topic + loss_worthiness

                        loss_sum += loss
                        loss_count += 1
                        for idx, i in enumerate(out_common):
                            if torch.argmax(i) == torch.argmax(y_topic_cur, dim=1)[idx]:
                                correct_topic += 1
                            total += 1
                        for idx, i in enumerate(out_worthiness):
                            if torch.argmax(i) == torch.argmax(y_worthiness_cur, dim=1)[idx]:
                                correct_worthiness += 1

                print('val_topic_acc:', correct_topic / total,
                      'val_worthiness_acc:', correct_worthiness / total,
                      'val_avg_loss:', loss_sum / loss_count)

            # es part
            if es_dif is not None and epoch != 0:
                if tol_epochs != 0:  # already in tolerance mode
                    if prev_train_loss - avg_train_loss > es_dif:  # leave tolerance mode
                        tol_epochs = 0
                    elif tol_epochs >= es_tol:  # tolerance limit exceeded
                        return
                    else:  # continue tolerance mode
                        tol_epochs += 1
                elif prev_train_loss - avg_train_loss <= es_dif:  # not in tolerance but to slow learning
                    if es_tol == 0:  # no tolerance
                        return
                    else:  # enter tolerance mode
                        tol_epochs += 1
            prev_train_loss = avg_train_loss

    def predict(self, X, with_dropout=False):
        self.model.eval()
        if with_dropout:
            for m in self.model.modules():
                if isinstance(m, nn.modules.Dropout):
                    m.train()

        x_img_t = torch.tensor(X[0])
        x_txt_t = torch.tensor(X[1])

        ds = TensorDataset(x_img_t, x_txt_t)
        dataloader = DataLoader(ds, batch_size=BATCH_SIZE)

        predictions_common = torch.tensor([])
        predictions_img = torch.tensor([])
        predictions_txt = torch.tensor([])
        predictions_worthiness = torch.tensor([])

        with torch.no_grad():
            for x_img_cur, x_txt_cur in dataloader:
                out_common, out_img, out_txt, out_worthiness = self.model(x_img_cur.float(), x_txt_cur.float())
                predictions_common = torch.cat((predictions_common, out_common), 0)
                predictions_img = torch.cat((predictions_img, out_img), 0)
                predictions_txt = torch.cat((predictions_txt, out_txt), 0)
                predictions_worthiness = torch.cat((predictions_worthiness, out_worthiness), 0)

        return predictions_common.numpy(), predictions_img.numpy(), predictions_txt.numpy(), predictions_worthiness.numpy()

    def predict_proba(self, X, **predict_kwargs):
        y_predicted_common, y_predicted_img, y_predicted_txt, y_predicted_worthiness = self.predict(X, **predict_kwargs)
        return np.exp(y_predicted_common), np.exp(y_predicted_img), np.exp(y_predicted_txt), np.exp(y_predicted_worthiness)

    def evaluate(self, X, y, verbose=0):
        y_topic_predicted, _, _, y_worthiness_predicted = self.predict(X)
        y_topic_target, y_worthiness_target = y
        return calculate_metrics(
            y_topic_predicted=y_topic_predicted,
            y_worthiness_predicted=y_worthiness_predicted,
            y_topic_target=y_topic_target,
            y_worthiness_target=y_worthiness_target,
            verbose=verbose
        )
