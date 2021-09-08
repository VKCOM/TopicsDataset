import scipy
import torch
from scipy import stats
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np

IMG_LEN = 1024
TXT_LEN = 300
BATCH_SIZE = 512


def prepare_train_loader(X, y, cur_batch_size=None):
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

    if cur_batch_size is None:
        cur_batch_size = BATCH_SIZE + (len(x_img) % BATCH_SIZE) // (len(x_img) // BATCH_SIZE)
        cur_batch_size += 2 if not cur_batch_size % 2 else 1
    print('cur batch size:', cur_batch_size)
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


class TopicsDecorator:
    """
    implies that X == [x_img, x_txt], where x_img and x_txt are numpy arrays
    """

    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    # implied that early stopping is only on train loss
    def fit(self, X, y, epochs=1, validation_data=None, weight_norm=False, es_dif=None, es_tol=0, verbose=0):

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

            margin_sum = 0.0
            margin_count = 0
            for x_img_cur, x_txt_cur, y_cur in train_loader:

                self.optimizer.zero_grad()
                output = self.model(x_img_cur, x_txt_cur)

                part = np.partition(-np.exp(output.detach().numpy()), 1, axis=1)
                margin = - part[:, 0] + part[:, 1]
                margin = torch.tensor(margin.reshape(-1, 1)).float()
                margin_sum += sum(margin)
                margin_count += margin.shape[0]

                if not weight_norm:
                    train_loss = F.nll_loss(output, torch.argmax(y_cur, dim=1))
                else:
                    if verbose != 0:
                        print('use weighted loss')
                    loss_non_reducted = F.nll_loss(output, torch.argmax(y_cur, dim=1), reduction='none')

                    output_detached = output.detach()
                    predictions = torch.exp(output_detached)
                    weights = torch.t(torch.tensor(stats.entropy(torch.t(predictions))))

                    norm_loss = loss_non_reducted * weights
                    norm_loss /= sum(norm_loss) / sum(loss_non_reducted)

                    assert abs(sum(norm_loss) - sum(loss_non_reducted)) < 1e-2

                    train_loss = torch.mean(norm_loss)

                train_loss.backward()
                train_loss_sum += train_loss

                train_loss_count += 1

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

            avg_train_loss = train_loss_sum / train_loss_count
            if verbose != 0:
                print('avg margin:', margin_sum/margin_count)
                print('epoch:', epoch, 'train_loss:', train_loss, 'average train loss', avg_train_loss)

            if validation_data is not None:
                self.model.eval()

                correct = 0
                total = 0
                val_loss_sum = 0.0
                val_loss_count = 0

                with torch.no_grad():
                    for x_img_cur, x_txt_cur, y_cur in val_loader:
                        output = self.model(x_img_cur, x_txt_cur)
                        val_loss = F.nll_loss(output, torch.argmax(y_cur, dim=1))
                        val_loss_sum += val_loss
                        val_loss_count += 1
                        for idx, i in enumerate(output):
                            if torch.argmax(i) == torch.argmax(y_cur, dim=1)[idx]:
                                correct += 1
                            total += 1
                if verbose != 0:
                    print('val_acc:', correct / total, 'val_avg_loss:', val_loss_sum / val_loss_count)

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
                outputs = self.model(x_img_cur.float(), x_txt_cur.float())
                predictions = torch.cat((predictions, outputs), 0)

        return predictions.numpy()

    def predict_proba(self, X, **predict_kwargs):
        y_predicted = self.predict(X, **predict_kwargs)
        return np.exp(y_predicted)

    def evaluate(self, X, y, verbose=1):
        y_predicted = self.predict(X)
        loss = F.nll_loss(torch.from_numpy(y_predicted), torch.argmax(torch.from_numpy(y), dim=1))

        correct = 0.0
        total = 0.0
        y_predicted_non_cat = y_predicted.argmax(axis=1)
        y_true_non_cat = y.argmax(axis=1)

        for i in range(y_predicted.shape[0]):
            if y_predicted_non_cat[i] == y_true_non_cat[i]:
                correct += 1
            total += 1

        if verbose != 0:
            print('val acc', correct / total)

        return loss, correct / total


# copy from https://github.com/seominseok0429/Learning-Loss-for-Active-Learning-Pytorch
class MarginRankingLearningLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MarginRankingLearningLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        random = torch.randperm(inputs.size(0))
        pred_loss = inputs[random]
        pred_lossi = inputs[:inputs.size(0) // 2]
        pred_lossj = inputs[inputs.size(0) // 2:]
        target_loss = targets.reshape(inputs.size(0), 1)
        target_loss = target_loss[random]
        target_lossi = target_loss[:inputs.size(0) // 2]
        target_lossj = target_loss[inputs.size(0) // 2:]
        final_target = torch.sign(target_lossi - target_lossj)

        return F.margin_ranking_loss(pred_lossi, pred_lossj, final_target, margin=self.margin, reduction='mean')


loss_pred_criterion = MarginRankingLearningLoss()


class LearningLossDecorator:
    def __init__(self, decorated_model, ll_model, ll_optimizer, ll_version=1):
        self.decorated_model = decorated_model
        self.ll_model = ll_model
        self.ll_optimizer = ll_optimizer
        self.ll_version = ll_version
        self.correllation_history = []

    def fit(self, X, y, epochs=1, verbose=0, **decorated_model_fit_kwargs):
        if verbose != 0:
            print('fit learning loss decorator on', X[0].shape[0], 'objects')

        train_loader = prepare_train_loader(X, y)

        for epoch in range(epochs):
            if verbose != 0:
                print('fit learning loss decorator, epoch', epoch)

            self.ll_model.train()
            self.decorated_model.model.eval()

            loss_loss_sum = 0.0
            loss_loss_count = 0

            for x_img_cur, x_txt_cur, y_cur in train_loader:

                self.ll_optimizer.zero_grad()
                with torch.no_grad():
                    output = self.decorated_model.model(x_img_cur, x_txt_cur).detach()
                actual_loss = F.nll_loss(output, torch.argmax(y_cur, dim=1), reduction='none')
                if self.ll_version == 3:
                    predicted_loss = self.ll_model(x_img_cur, x_txt_cur)
                else:
                    predicted_loss = self.ll_model(output)

                corr, p_val = scipy.stats.spearmanr(actual_loss.detach().numpy(), predicted_loss.detach().numpy())
                print('spearman:', corr, 'p value_', p_val)
                self.correllation_history.append((corr, p_val))

                if self.ll_version == 4:
                    loss_loss = loss_pred_criterion(predicted_loss, actual_loss)
                else:
                    loss_loss = F.mse_loss(predicted_loss, actual_loss.view(-1, 1))

                loss_loss_sum += loss_loss
                loss_loss_count += 1

                loss_loss.backward()
                self.ll_optimizer.step()

            self.decorated_model.fit(X, y, epochs=1, verbose=verbose, **decorated_model_fit_kwargs)

    def predict(self, X, with_dropout=False):
        return self.decorated_model.predict(X, with_dropout)

    def predict_proba(self, X, **predict_kwargs):
        return self.decorated_model.predict_proba(X, **predict_kwargs)

    def evaluate(self, X, y, verbose=1):
        return self.decorated_model.evaluate(X, y, verbose)

    def predict_loss(self, X):
        print('predicting loss')
        self.decorated_model.model.eval()
        self.ll_model.eval()

        dataloader = prepare_predict_loader(X)
        predicted_losses = torch.tensor([])
        with torch.no_grad():
            for x_img_cur, x_txt_cur in dataloader:
                if self.ll_version == 3:
                    losses = self.ll_model(x_img_cur, x_txt_cur)
                else:
                    predictions = self.decorated_model.model(x_img_cur.float(), x_txt_cur.float()).detach()
                    losses = self.ll_model(predictions)
                predicted_losses = torch.cat((predicted_losses, losses), 0)

        return predicted_losses.numpy()


class TridentDecorator:
    """
    implies that X == [x_img, x_txt], where x_img and x_txt are numpy arrays
    """

    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, X, y, epochs=1, validation_data=None, es_dif=None, es_tol=0, verbose=1, use_batch_norm=True, batch_size=None):
        if verbose != 0:
            print('fit on', X[0].shape[0], 'objects')
        train_loader = prepare_train_loader(X, y, cur_batch_size=batch_size)

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

            for x_img_cur, x_txt_cur, y_cur in train_loader:

                self.optimizer.zero_grad()
                out_common, out_img, out_txt = self.model(x_img_cur, x_txt_cur)
                target = torch.argmax(y_cur, dim=1)

                loss_common = F.nll_loss(out_common, target)
                loss_img = F.nll_loss(out_img, target)
                loss_txt = F.nll_loss(out_txt, target)
                loss = (loss_common + loss_img + loss_txt) / 3.0

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

                correct = 0
                total = 0
                loss_sum = 0.0
                loss_count = 0

                with torch.no_grad():
                    for x_img_cur, x_txt_cur, y_cur in val_loader:
                        out_common, _, _ = self.model(x_img_cur, x_txt_cur)
                        loss = F.nll_loss(out_common, torch.argmax(y_cur, dim=1))
                        loss_sum += loss
                        loss_count += 1
                        for idx, i in enumerate(out_common):
                            if torch.argmax(i) == torch.argmax(y_cur, dim=1)[idx]:
                                correct += 1
                            total += 1

                print('val_acc:', correct / total, 'val_avg_loss:', loss_sum / loss_count)

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

        with torch.no_grad():
            for x_img_cur, x_txt_cur in dataloader:
                out_common, out_img, out_txt = self.model(x_img_cur.float(), x_txt_cur.float())
                predictions_common = torch.cat((predictions_common, out_common), 0)
                predictions_img = torch.cat((predictions_img, out_img), 0)
                predictions_txt = torch.cat((predictions_txt, out_txt), 0)

        return predictions_common.numpy(), predictions_img.numpy(), predictions_txt.numpy()

    def predict_proba(self, X, **predict_kwargs):
        y_predicted_common, y_predicted_img, y_predicted_txt = self.predict(X, **predict_kwargs)
        return np.exp(y_predicted_common), np.exp(y_predicted_img), np.exp(y_predicted_txt)

    def evaluate(self, X, y, verbose=0):
        y_predicted_common, _, _ = self.predict(X)
        loss = F.nll_loss(torch.from_numpy(y_predicted_common), torch.argmax(torch.from_numpy(y), dim=1))

        correct = 0.0
        total = 0.0
        y_predicted_non_cat = y_predicted_common.argmax(axis=1)
        y_true_non_cat = y.argmax(axis=1)

        for i in range(y_predicted_common.shape[0]):
            if y_predicted_non_cat[i] == y_true_non_cat[i]:
                correct += 1
            total += 1

        print('val acc', correct / total)

        return loss, correct / total


class TridentMTLDecorator:
    """
    implies that X == [x_img, x_txt], where x_img and x_txt are numpy arrays
    """

    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, X, y, epochs=1, validation_data=None, es_dif=None, es_tol=0, verbose=1, use_batch_norm=True):
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

            for x_img_cur, x_txt_cur, y_cur in train_loader:

                self.optimizer.zero_grad()
                trainable_loss, raw_losses = self.model(x_img_cur, x_txt_cur, y_cur)
                loss_common, loss_img, loss_txt = raw_losses
                trainable_loss.backward()
                loss_sum += (loss_common + loss_img + loss_txt) / 3.0
                loss_count += 1

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

            avg_train_loss = loss_sum / loss_count
            print('epoch:', epoch, 'train_loss:', (loss_common + loss_img + loss_txt) / 3.0, 'average train loss', loss_sum / loss_count)

            if validation_data is not None:
                self.model.eval()

                correct = 0
                total = 0
                loss_sum = 0.0
                loss_count = 0

                with torch.no_grad():
                    for x_img_cur, x_txt_cur, y_cur in val_loader:
                        out_common, _, _ = self.model.model(x_img_cur, x_txt_cur)
                        loss = F.nll_loss(out_common, torch.argmax(y_cur, dim=1))
                        loss_sum += loss
                        loss_count += 1
                        for idx, i in enumerate(out_common):
                            if torch.argmax(i) == torch.argmax(y_cur, dim=1)[idx]:
                                correct += 1
                            total += 1

                print('val_acc:', correct / total, 'val_avg_loss:', loss_sum / loss_count)

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

    def predict(self, X, with_dropout=False, **kwargs):

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

        with torch.no_grad():
            for x_img_cur, x_txt_cur in dataloader:
                out_common, out_img, out_txt = self.model.model(x_img_cur.float(), x_txt_cur.float())
                predictions_common = torch.cat((predictions_common, out_common), 0)
                predictions_img = torch.cat((predictions_img, out_img), 0)
                predictions_txt = torch.cat((predictions_txt, out_txt), 0)

        return predictions_common.numpy(), predictions_img.numpy(), predictions_txt.numpy()

    def predict_proba(self, X, **predict_kwargs):
        y_predicted_common, y_predicted_img, y_predicted_txt = self.predict(X, **predict_kwargs)
        return np.exp(y_predicted_common), np.exp(y_predicted_img), np.exp(y_predicted_txt)

    def evaluate(self, X, y, verbose=0):
        y_predicted_common, _, _ = self.predict(X)
        loss = F.nll_loss(torch.from_numpy(y_predicted_common), torch.argmax(torch.from_numpy(y), dim=1))

        correct = 0.0
        total = 0.0
        y_predicted_non_cat = y_predicted_common.argmax(axis=1)
        y_true_non_cat = y.argmax(axis=1)

        for i in range(y_predicted_common.shape[0]):
            if y_predicted_non_cat[i] == y_true_non_cat[i]:
                correct += 1
            total += 1

        print('val acc', correct / total)

        return loss, correct / total


class TridentAsNormDecorator(TridentDecorator):
    """
    implies that X == [x_img, x_txt], where x_img and x_txt are numpy arrays
    """
    def predict(self, X, with_dropout=False, **kwargs):

        if with_dropout:
            self.model.train()
        else:
            self.model.eval()

        x_img_t = torch.tensor(X[0])
        x_txt_t = torch.tensor(X[1])

        ds = TensorDataset(x_img_t, x_txt_t)
        dataloader = DataLoader(ds, batch_size=BATCH_SIZE)

        predictions_common = torch.tensor([])

        with torch.no_grad():
            for x_img_cur, x_txt_cur in dataloader:
                out_common, _, _ = self.model(x_img_cur.float(), x_txt_cur.float())
                predictions_common = torch.cat((predictions_common, out_common), 0)

        return predictions_common.numpy()

    def predict_proba(self, X, **predict_kwargs):
        y_predicted_common = self.predict(X, **predict_kwargs)
        return np.exp(y_predicted_common)

    def evaluate(self, X, y, verbose=0):
        y_predicted_common = self.predict(X)
        loss = F.nll_loss(torch.from_numpy(y_predicted_common), torch.argmax(torch.from_numpy(y), dim=1))

        correct = 0.0
        total = 0.0
        y_predicted_non_cat = y_predicted_common.argmax(axis=1)
        y_true_non_cat = y.argmax(axis=1)

        for i in range(y_predicted_common.shape[0]):
            if y_predicted_non_cat[i] == y_true_non_cat[i]:
                correct += 1
            total += 1

        print('val acc', correct / total)

        return loss, correct / total


class TridentMTLAsNormDecorator(TridentMTLDecorator):
    """
    implies that X == [x_img, x_txt], where x_img and x_txt are numpy arrays
    """
    def predict(self, X, with_dropout=False, **kwargs):

        if with_dropout:
            self.model.train()
        else:
            self.model.eval()

        x_img_t = torch.tensor(X[0])
        x_txt_t = torch.tensor(X[1])

        ds = TensorDataset(x_img_t, x_txt_t)
        dataloader = DataLoader(ds, batch_size=BATCH_SIZE)

        predictions_common = torch.tensor([])

        with torch.no_grad():
            for x_img_cur, x_txt_cur in dataloader:
                out_common, _, _ = self.model.model(x_img_cur.float(), x_txt_cur.float())
                predictions_common = torch.cat((predictions_common, out_common), 0)

        return predictions_common.numpy()

    def predict_proba(self, X, **predict_kwargs):
        y_predicted_common = self.predict(X, **predict_kwargs)
        return np.exp(y_predicted_common)

    def evaluate(self, X, y, verbose=0):
        y_predicted_common = self.predict(X)
        loss = F.nll_loss(torch.from_numpy(y_predicted_common), torch.argmax(torch.from_numpy(y), dim=1))

        correct = 0.0
        total = 0.0
        y_predicted_non_cat = y_predicted_common.argmax(axis=1)
        y_true_non_cat = y.argmax(axis=1)

        for i in range(y_predicted_common.shape[0]):
            if y_predicted_non_cat[i] == y_true_non_cat[i]:
                correct += 1
            total += 1

        print('val acc', correct / total)

        return loss, correct / total
