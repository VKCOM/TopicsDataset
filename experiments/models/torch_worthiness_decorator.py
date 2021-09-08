import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

IMG_LEN = 1024
TXT_LEN = 300
BATCH_SIZE = 512


def prepare_train_loader(X, y):
    x_img = X[0]
    x_txt = X[1]

    p = np.random.permutation(len(x_img))
    x_img = x_img[p]
    x_txt = x_txt[p]
    y = y[p]

    x_img_train_t = torch.tensor(x_img).float()
    x_txt_train_t = torch.tensor(x_txt).float()
    y_train_t = torch.tensor(y)

    train_ds = TensorDataset(x_img_train_t, x_txt_train_t, y_train_t)

    cur_batch_size = BATCH_SIZE + (len(x_img) % BATCH_SIZE) // (len(x_img) // BATCH_SIZE)
    cur_batch_size += 2 if not cur_batch_size % 2 else 1

    train_loader = DataLoader(train_ds, batch_size=cur_batch_size)
    return train_loader


def prepare_val_loader(validation_data):
    x_img_val_t = torch.tensor(validation_data[0][0]).float()
    x_txt_val_t = torch.tensor(validation_data[0][1]).float()
    y_val_t = torch.tensor(validation_data[1])
    val_ds = TensorDataset(x_img_val_t, x_txt_val_t, y_val_t)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    return val_loader


def prepare_predict_loader(X):
    x_img_t = torch.tensor(X[0]).float()
    x_txt_t = torch.tensor(X[1]).float()
    ds = TensorDataset(x_img_t, x_txt_t)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    return loader


def calculate_metrics(y_predicted, y_target, verbose=1):
    loss = F.nll_loss(
        torch.from_numpy(y_predicted),
        torch.argmax(torch.from_numpy(y_target), dim=1)
    )

    correct = 0

    total = 0
    y_predicted_non_cat = y_predicted.argmax(axis=1)
    y_target_non_cat = y_target.argmax(axis=1)

    for i in range(y_predicted.shape[0]):
        if y_predicted_non_cat[i] == y_target_non_cat[i]:
            correct += 1
        total += 1

    worthiness_roc_auc_score = roc_auc_score(y_target, y_predicted)
    precision, recall, threshold = precision_recall_curve(
        np.argmax(y_target, axis=1),
        np.argmax(y_predicted, axis=1)
    )
    worthiness_pr_auc_score = auc(recall, precision)

    if verbose != 0:
        print('val acc', correct / total)
        print('roc auc score', worthiness_roc_auc_score)
        print('pr auc score', worthiness_pr_auc_score)

    return {
        'loss': loss,
        'accuracy': correct / total,
        'roc_auc_score': worthiness_roc_auc_score,
        'pr_auc_score': worthiness_pr_auc_score
    }


class WorthinessDecorator:
    """
    implies that
    X == [x_img, x_txt], where x_img and x_txt are numpy arrays
    y == y_worthiness, y_worthiness - numpy array
    """

    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    # implied that early stopping is only on train loss
    def fit(self, X, y, epochs=1, validation_data=None, es_dif=None, es_tol=0, verbose=0, weight=None):

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

            for x_img_cur, x_txt_cur, y_cur in train_loader:

                self.optimizer.zero_grad()
                output = self.model(x_img_cur, x_txt_cur)

                train_loss = F.nll_loss(output, torch.argmax(y_cur, dim=1), weight=weight)
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

                correct = 0
                total = 0
                val_loss_sum = 0.0
                val_loss_count = 0
                predictions = torch.tensor([])

                with torch.no_grad():
                    for x_img_cur, x_txt_cur, y_cur in val_loader:
                        output = self.model(x_img_cur, x_txt_cur)
                        predictions = torch.cat((predictions, output), 0)
                        val_loss = F.nll_loss(output, torch.argmax(y_cur, dim=1), weight=weight)

                        val_loss_sum += val_loss
                        val_loss_count += 1
                        for idx, i in enumerate(output):
                            if torch.argmax(i) == torch.argmax(y_cur, dim=1)[idx]:
                                correct += 1
                            total += 1

                predictions = predictions.numpy()
                precision, recall, threshold = precision_recall_curve(
                    np.argmax(validation_data[1], axis=1),
                    np.argmax(predictions, axis=1)
                )
                worthiness_pr_auc_score = auc(recall, precision)
                if verbose != 0:
                    print('val_acc:', correct / total,
                          'val_avg_loss:', val_loss_sum / val_loss_count,
                          'roc auc score:', roc_auc_score(validation_data[1], predictions),
                          'pr auc score:', worthiness_pr_auc_score)

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
        predictions = torch.tensor([])

        with torch.no_grad():
            for x_img_cur, x_txt_cur in dataloader:
                output = self.model(x_img_cur.float(), x_txt_cur.float())
                predictions = torch.cat((predictions, output), 0)

        return predictions.numpy()

    def predict_proba(self, X, **predict_kwargs):
        y_predicted = self.predict(X, **predict_kwargs)
        return np.exp(y_predicted)

    def evaluate(self, X, y, verbose=1):
        y_predicted = self.predict(X)
        return calculate_metrics(
            y_predicted=y_predicted,
            y_target=y,
        )
