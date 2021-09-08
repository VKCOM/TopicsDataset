import time

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


IMG_LEN = 1024
TXT_LEN = 300
N_CLASSES = 50
BATCH_SIZE = 2048
criterion = nn.MSELoss()


def prepare_data_for_torch(X_train, X_val):
    x_img_train, x_txt_train = X_train[0], X_train[1]
    x_img_val, x_txt_val = X_val[0], X_val[1]

    x_img_train_t = torch.tensor(x_img_train).float()
    x_img_val_t = torch.tensor(x_img_val).float()

    x_txt_train_t = torch.tensor(x_txt_train).float()
    x_txt_val_t = torch.tensor(x_txt_val).float()

    train_ds = TensorDataset(x_img_train_t, x_txt_train_t)
    val_ds = TensorDataset(x_img_val_t, x_txt_val_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    return train_loader, val_loader


def fit_autoencoder(autoencoder, optimizer, epochs, X_train, X_val, verbose=1):
    train_loader, val_loader = prepare_data_for_torch(X_train, X_val)

    train_img_loss_history = []
    train_txt_loss_history = []

    val_img_loss_history = []
    val_txt_loss_history = []

    start_time = time.time()

    for epoch in range(epochs):
        autoencoder.train()

        loss_img_sum = 0.0
        loss_txt_sum = 0.0
        loss_sum = 0.0
        loss_count = 0

        for x_img_cur, x_txt_cur in train_loader:
            autoencoder.zero_grad()
            out_img, out_txt = autoencoder(inp_img=x_img_cur, inp_txt=x_txt_cur)
            loss_img = criterion(out_img, x_img_cur)
            loss_txt = criterion(out_txt, x_txt_cur)
            loss = loss_img + loss_txt

            loss_img_sum += loss_img
            loss_txt_sum += loss_txt
            loss_sum += loss
            loss_count += 1

            loss.backward()
            optimizer.step()

        if verbose != 0:
            print(
                'epoch:', epoch,
                'train img loss:', "%.3f" % (loss_img_sum / loss_count).item(),
                'txt_loss:', "%.3f" % (loss_txt_sum / loss_count).item(),
                'img + txt loss', "%.3f" % (loss_sum / loss_count).item()
            )
        train_img_loss_history.append((loss_img_sum / loss_count).item())
        train_txt_loss_history.append((loss_txt_sum / loss_count).item())

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

        if verbose != 0:
            print(
                'val img loss:', "%.3f" % (val_loss_img_sum / val_loss_count).item(),
                'val txt_loss:', "%.3f" % (val_loss_txt_sum / val_loss_count).item(),
                'img + txt loss', "%.3f" % (val_loss_sum / val_loss_count).item()
            )
        val_img_loss_history.append((val_loss_img_sum / val_loss_count).item())
        val_txt_loss_history.append((val_loss_txt_sum / val_loss_count).item())

    operation_time = time.time() - start_time

    if verbose != 0:
        print('autoencoder fitting finished for', operation_time, 'seconds')

    return train_img_loss_history, train_txt_loss_history, val_img_loss_history, val_txt_loss_history, operation_time


class Encoder(nn.Module):
    def __init__(self, d, drop=0.5):
        super().__init__()
        self.fc_img_1 = nn.Linear(IMG_LEN, d * 4)
        self.bn_img_1 = nn.BatchNorm1d(num_features=d * 4)
        self.fc_img_2 = nn.Linear(d * 4, d * 2)
        self.bn_img_2 = nn.BatchNorm1d(num_features=d * 2)

        self.fc_txt_1 = nn.Linear(TXT_LEN, d * 2)
        self.bn_txt_1 = nn.BatchNorm1d(num_features=d * 2)
        self.fc_txt_2 = nn.Linear(d * 2, d * 2)
        self.bn_txt_2 = nn.BatchNorm1d(num_features=d * 2)

        self.fc = nn.Linear(d * 4, d)
        self.bn = nn.BatchNorm1d(num_features=d)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, inp_img, inp_txt):
        x_img = self.dropout(self.bn_img_1(F.relu(self.fc_img_1(inp_img))))
        x_img = self.dropout(self.bn_img_2(F.relu(self.fc_img_2(x_img))))

        x_txt = self.dropout(self.bn_txt_1(F.relu(self.fc_txt_1(inp_txt))))
        x_txt = self.dropout(self.bn_txt_2(F.relu(self.fc_txt_2(x_txt))))

        x = torch.cat((x_img, x_txt), 1)
        x = self.dropout(self.bn(F.relu(self.fc(x))))
        return x


class Decoder(nn.Module):
    def __init__(self, d, drop=0.5):
        super().__init__()

        self.fc_img_1 = nn.Linear(d, 4 * d)
        self.fc_img_2 = nn.Linear(4 * d, IMG_LEN)

        self.fc_txt_1 = nn.Linear(d, 2 * d)
        self.fc_txt_2 = nn.Linear(2 * d, TXT_LEN)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, x):
        x_img = self.dropout(F.relu(self.fc_img_1(x)))
        x_img = self.fc_img_2(x_img)

        x_txt = self.dropout(F.relu(self.fc_txt_1(x)))
        x_txt = self.fc_txt_2(x_txt)

        return x_img, x_txt


class Autoencoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = Encoder(d)
        self.decoder = Decoder(d)

    def forward(self, inp_img, inp_txt):
        x = self.encoder(inp_img, inp_txt)
        x_img, x_txt = self.decoder(x)
        return x_img, x_txt


class EncoderTrident(nn.Module):
    def __init__(self, d, drop=0.5):
        super().__init__()
        self.fc_img_1 = nn.Linear(IMG_LEN, d * 4)
        self.bn_img_1 = nn.BatchNorm1d(num_features=d * 4)
        self.fc_img_2 = nn.Linear(d * 4, d * 2)
        self.bn_img_2 = nn.BatchNorm1d(num_features=d * 2)

        self.fc_txt_1 = nn.Linear(TXT_LEN, d * 2)
        self.bn_txt_1 = nn.BatchNorm1d(num_features=d * 2)
        self.fc_txt_2 = nn.Linear(d * 2, d * 2)
        self.bn_txt_2 = nn.BatchNorm1d(num_features=d * 2)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, inp_img, inp_txt):
        x_img = self.dropout(self.bn_img_1(F.relu(self.fc_img_1(inp_img))))
        x_img = self.dropout(self.bn_img_2(F.relu(self.fc_img_2(x_img))))

        x_txt = self.dropout(self.bn_txt_1(F.relu(self.fc_txt_1(inp_txt))))
        x_txt = self.dropout(self.bn_txt_2(F.relu(self.fc_txt_2(x_txt))))

        return x_img, x_txt


class DecoderTrident(nn.Module):
    def __init__(self, d, drop=0.5):
        super().__init__()
        self.fc = nn.Linear(4 * d, 2 * d)

        self.fc_img_1 = nn.Linear(2 * d, 4 * d)
        self.fc_img_2 = nn.Linear(4 * d, IMG_LEN)

        self.fc_txt_1 = nn.Linear(2 * d, 2 * d)
        self.fc_txt_2 = nn.Linear(2 * d, TXT_LEN)

        self.dropout = nn.modules.Dropout(p=drop)

    def forward(self, x_img, x_txt):
        x = self.dropout(F.relu(self.fc(torch.cat((x_img, x_txt), 1))))

        x_img = self.dropout(F.relu(self.fc_img_1(x)))
        x_img = self.fc_img_2(x_img)

        x_txt = self.dropout(F.relu(self.fc_txt_1(x)))
        x_txt = self.fc_txt_2(x_txt)

        return x_img, x_txt


class AutoencoderTrident(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = EncoderTrident(d)
        self.decoder = DecoderTrident(d)

    def forward(self, inp_img, inp_txt):
        x_img, x_txt = self.encoder(inp_img, inp_txt)
        x_img, x_txt = self.decoder(x_img, x_txt)
        return x_img, x_txt
