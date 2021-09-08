import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 50


class LearningLossModel(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.fcs = [nn.Linear(N_CLASSES, N_CLASSES) for _ in range(n_hidden)]
        self.bns = [nn.BatchNorm1d(num_features=N_CLASSES) for _ in range(n_hidden)]
        self.out = nn.Linear(N_CLASSES, 1)
        self.dropout = nn.modules.Dropout(p=0.5)

    def forward(self, x):
        for fc, bn in zip(self.fcs, self.bns):
            x = self.dropout(bn(F.relu(fc(x))))

        x = self.out(x)
        return x


class LearningLossModel2(nn.Module):
    def __init__(self, n_hidden=1, n_hidden_raw=1, metrics=None, use_raw=True, d=8):
        if not use_raw and metrics is None:
            raise ValueError('at least metrics should be provided or use_raw set to true')

        super().__init__()

        self.metrics = metrics
        self.use_raw = use_raw

        if use_raw:
            self.fcs = [nn.Linear(N_CLASSES, N_CLASSES) for _ in range(n_hidden_raw)]
            self.bns = [nn.BatchNorm1d(num_features=N_CLASSES) for _ in range(n_hidden_raw)]
            self.raw_metric_fc = nn.Linear(N_CLASSES, 1)

        self.dropout = nn.modules.Dropout(p=0.5)

        input_count = (0 if metrics is None else len(metrics)) + (1 if use_raw else 0)

        self.metric_fc_1 = nn.Linear(input_count, d)
        self.metric_bn_1 = nn.BatchNorm1d(num_features=d)

        self.metric_fcs = [nn.Linear(d, d) for _ in range(n_hidden - 1)]
        self.metric_bns = [nn.BatchNorm1d(num_features=d) for _ in range(n_hidden - 1)]

        self.out = nn.Linear(d, 1)

    def forward(self, input):
        if self.use_raw:
            x_raw = input.detach()
            for fc, bn in zip(self.fcs, self.bns):
                x_raw = self.dropout(bn(F.relu(fc(input))))
            raw_metric = self.raw_metric_fc(x_raw)
            metric_vals = [raw_metric]
        else:
            metric_vals = []

        if self.metrics is not None:
            metric_vals.extend([metric(input.detach()) for metric in self.metrics])
            joined_metrics = torch.cat(tuple(metric_vals), dim=1)
        else:
            joined_metrics = metric_vals[0]

        x = self.dropout(self.metric_bn_1(F.relu(self.metric_fc_1(joined_metrics))))
        for fc, bn in zip(self.metric_fcs, self.metric_bns):
            x = self.dropout(bn(F.relu(fc(x))))
        result_metric = self.out(x)

        return result_metric


class LearningLossModel2_1(nn.Module):
    def __init__(self, n_hidden=1, metrics=None, use_raw=True, d=8):
        if not use_raw and metrics is None:
            raise ValueError('at least metrics should be provided or use_raw set to true')

        super().__init__()

        self.metrics = metrics
        self.use_raw = use_raw

        input_dim = 0

        if use_raw:
            input_dim += N_CLASSES
        if metrics is not None:
            input_dim += len(metrics)

        self.dropout = nn.modules.Dropout(p=0.5)

        self.fc_1 = nn.Linear(input_dim, d)
        self.bn_1 = nn.BatchNorm1d(num_features=d)

        self.fcs = [nn.Linear(d, d) for _ in range(n_hidden - 1)]
        self.bns = [nn.BatchNorm1d(num_features=d) for _ in range(n_hidden - 1)]

        self.out = nn.Linear(d, 1)

    def forward(self, input):
        if self.use_raw:
            x_raw = input.detach()
            metric_vals = [x_raw]
        else:
            metric_vals = []

        if self.metrics is not None:
            metric_vals.extend([metric(input.detach()) for metric in self.metrics])
            joined_metrics = torch.cat(tuple(metric_vals), dim=1)
        else:
            joined_metrics = metric_vals[0]

        x = self.dropout(self.bn_1(F.relu(self.fc_1(joined_metrics))))
        for fc, bn in zip(self.fcs, self.bns):
            x = self.dropout(bn(F.relu(fc(x))))
        result_metric = self.out(x)

        return result_metric


class LearningLossModel2_2(nn.Module):
    def __init__(self, metrics=None):
        super().__init__()

        self.metrics = metrics

        input_dim = 0

        if metrics is not None:
            input_dim += len(metrics)

        self.fc = nn.Linear(input_dim, 1)

    def forward(self, input):
        metric_vals = [metric(input.detach()) for metric in self.metrics]
        joined_metrics = torch.cat(tuple(metric_vals), dim=1)

        x = self.fc(joined_metrics)

        return x


class LearningLossModel3(nn.Module):
    def __init__(self, model, metrics, n_hidden=1, d=4):
        super().__init__()
        self.model = model
        self.metrics = metrics

        self.fc_0 = nn.Linear(len(metrics), d)
        self.bn_0 = nn.BatchNorm1d(num_features=d)

        self.fcs = [nn.Linear(d, d) for _ in range(n_hidden - 1)]
        self.bns = [nn.BatchNorm1d(num_features=d) for _ in range(n_hidden- 1)]

        self.out = nn.Linear(d, 1)
        self.dropout = nn.modules.Dropout(p=0.5)

    def forward(self, x_img, x_txt):
        metric_vals = [metric(self.model, x_img, x_txt) for metric in self.metrics]
        joined_metrics = torch.cat(tuple(metric_vals), dim=1)

        x = self.dropout(self.bn_0(F.relu(self.fc_0(joined_metrics))))
        for fc, bn in zip(self.fcs, self.bns):
            x = self.dropout(bn(F.relu(fc(x))))
        x = self.out(x)

        return x


class LearningLossModel3_1(nn.Module):
    def __init__(self, model, n_hidden=1, metrics=None, use_raw=True, d=8):
        super().__init__()
        self.model = model
        self.metrics = metrics
        self.use_raw = use_raw

        input_dim = 0
        if use_raw:
            input_dim += N_CLASSES
        if not metrics is None:
            input_dim += len(metrics)

        self.fc_0 = nn.Linear(input_dim, d)
        self.bn_0 = nn.BatchNorm1d(num_features=d)

        self.fcs = [nn.Linear(d, d) for _ in range(n_hidden - 1)]
        self.bns = [nn.BatchNorm1d(num_features=d) for _ in range(n_hidden- 1)]

        self.out = nn.Linear(d, 1)
        self.dropout = nn.modules.Dropout(p=0.5)

    def get_params(self):
        res = list(self.fc_0.parameters()) + list(self.bn_0.parameters())
        for fc, bn in zip(self.fcs, self.bns):
            res += list(fc.parameters())
            res += list(bn.parameters())
        res += self.out.parameters()
        return res

    def forward(self, x_img, x_txt):
        if self.use_raw:
            self.model.eval()
            with torch.no_grad():
                metric_vals = [self.model(x_img, x_txt).detach()]
        else:
            metric_vals = []

        if self.metrics is not None:
            metric_vals.extend([metric(self.model, x_img, x_txt) for metric in self.metrics])

        joined_metrics = torch.cat(tuple(metric_vals), dim=1)

        x = self.dropout(self.bn_0(F.relu(self.fc_0(joined_metrics))))
        for fc, bn in zip(self.fcs, self.bns):
            x = self.dropout(bn(F.relu(fc(x))))
        x = self.out(x)

        return x