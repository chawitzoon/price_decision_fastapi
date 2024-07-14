import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from copy import deepcopy as dc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import joblib


# custom class to data into model training
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, device="cpu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_stacked_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

        self.device = device

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(
            self.device
        )
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(
            self.device
        )

        # Pass input through LSTM layers
        out, _ = self.lstm(x, (h0, c0))

        # Take the last time step's output
        out = self.fc(out[:, -1, :])
        return out


def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


# data preparation for training
# now it is offline csv file only. later do recreating file from retreived api data and update in s3
def load_data(filepath):
    # csv file path
    data = pd.read_csv(filepath)
    return data


def get_scaler(data_np):
    scaler = StandardScaler()
    data_scaler = scaler.fit(data_np)
    return data_scaler


def preprocess_data(data, lookback_days, train_test_split=None):
    # preprocess and add columns for lstm
    data = data[["Date", "Close"]]
    data["Date"] = pd.to_datetime(data["Date"])
    shifted_df = prepare_dataframe_for_lstm(data, lookback_days)
    shifted_df_to_np = shifted_df.to_numpy()
    X_np = shifted_df_to_np[:, 1:]
    y_np = shifted_df_to_np[:, 0].reshape((-1, 1))

    # get scaler and scale it
    X_scaler = get_scaler(X_np)
    y_scaler = get_scaler(y_np)

    # Save the Scalers
    joblib.dump(X_scaler, "mlib_model/X_scaler.pkl")
    joblib.dump(y_scaler, "mlib_model/y_scaler.pkl")

    X = X_scaler.transform(X_np)
    y = y_scaler.transform(y_np)

    X = dc(np.flip(X, axis=1))

    # train-test split (no evaluation set yet)
    split_index = int(len(X) * train_test_split)
    X_train = X[:split_index].reshape((-1, lookback_days, 1))
    X_test = X[split_index:].reshape((-1, lookback_days, 1))

    y_train = y[:split_index].reshape((-1, 1))
    y_test = y[split_index:].reshape((-1, 1))

    return X_train, X_test, y_train, y_test, X_scaler, y_scaler


def to_torch(value_np):
    # convert to torch tensor
    return torch.tensor(value_np).float()


def to_loader(X, y, batch_size=16, train=False):
    dataset = TimeSeriesDataset(X, y)
    # convert to loader, used in training iteration
    if train:
        shuffle = True
    else:
        shuffle = False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index("Date", inplace=True)

    for i in range(1, n_steps + 1):
        df[f"Close(t-{i})"] = df["Close"].shift(i)

    df.dropna(inplace=True)

    return df


class Agent:
    def __init__(
        self,
        input_size,
        hidden_size,
        num_stacked_layers,
        device="cpu",
        learning_rate=0.001,
        num_epochs=10,
    ):
        self.model = LSTM(input_size, hidden_size, num_stacked_layers, device)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.device = device
        self.model.to(self.device)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # default optimizer and loss funtion is set in Agent internally. We may assign custom ones when Agent object is initiated
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.loss_function = nn.MSELoss()

        # must specify train_loader, test_loader, X_scaler, y_scaler right after Agent object is initiated
        self.train_loader = None
        self.test_loader = None
        self.X_scaler = None
        self.y_scaler = None

    def train_one_epoch(self, epoch):
        self.model.train(True)
        print(f"Epoch: {epoch + 1}")
        running_loss = 0.0

        for batch_index, batch in enumerate(self.train_loader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            output = self.model(x_batch)
            loss = self.loss_function(output, y_batch)
            running_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print(
                    "Batch {0}, Loss: {1:.3f}".format(
                        batch_index + 1, avg_loss_across_batches
                    )
                )
                running_loss = 0.0

    def validate_one_epoch(self):
        self.model.train(False)
        running_loss = 0.0

        for _, batch in enumerate(self.test_loader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            with torch.no_grad():
                output = self.model(x_batch)
                loss = self.loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(self.test_loader)

        print("Val Loss: {0:.3f}".format(avg_loss_across_batches))
        print("***************************************************")
        print()

    def train(self):
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            self.validate_one_epoch()

    def visualize(self, X, y):
        # visual now in before rescaling
        with torch.no_grad():
            predicted = self.model(X.to(self.device)).to("cpu").numpy()
            train_predictions = predicted.flatten()
        plt.plot(y, label="Actual Close")
        plt.plot(train_predictions, label="Predicted Close")
        plt.xlabel("Day")
        plt.ylabel("Close")
        plt.legend()
        plt.show()

    def predict(self, X):
        # X should be np size (batch_size, lookback_days) such as (6509, 7)
        X = to_torch(np.expand_dims(self.X_scaler.transform(X), -1))

        y_predicted = self.model(X.to(self.device)).detach().cpu().numpy().flatten()
        y = self.y_scaler.inverse_transform(y_predicted.reshape(-1, 1))
        return y

    def save_model_weight(self, filename="mlib_model/lstm_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def save_hyper_param(self, filename="mlib_model/lstm_model_hyper_param.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_size": self.input_size,
                    "hidden_size": self.hidden_size,
                    "num_stacked_layers": self.num_stacked_layers,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )
