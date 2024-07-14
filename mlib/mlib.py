"""ML prediction Library"""
from mlib.mlib_util import get_device, to_torch, LSTM
import numpy as np
import torch
import json
import joblib
import logging

logging.basicConfig(level=logging.INFO)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

base_dir = Path(__file__).resolve().parent

MODEL_WEIGHT_FILEPATH = base_dir / "mlib_model" / "lstm_model.pth"
HYPER_PARAM_FILEPATH = base_dir / "mlib_model" / "lstm_model_hyper_param.json"
X_SCALER_FILEPATH = base_dir / "mlib_model" / "X_scaler.pkl"
Y_SCALER_FILEPATH = base_dir / "mlib_model" / "y_scaler.pkl"


def load_model(
    model_weight_filepath=MODEL_WEIGHT_FILEPATH,
    hyper_param_filepath=HYPER_PARAM_FILEPATH,
):
    """Grabs model from disk"""
    hyper_param = load_hyper_param(hyper_param_filepath)

    # Load the model
    loaded_model = LSTM(
        hyper_param["input_size"],
        hyper_param["hidden_size"],
        hyper_param["num_stacked_layers"],
    )
    loaded_model.load_state_dict(torch.load(model_weight_filepath))
    # Set the model to evaluation mode
    loaded_model.eval()
    return loaded_model


def load_hyper_param(filepath=HYPER_PARAM_FILEPATH):
    # Load param info from JSON file
    with open(filepath, "r") as f:
        hyper_param = json.load(f)
    return hyper_param


def load_scaler(X_scaler=X_SCALER_FILEPATH, y_scaler=Y_SCALER_FILEPATH):
    X_scaler = joblib.load(X_scaler)
    y_scaler = joblib.load(y_scaler)
    return X_scaler, y_scaler


def predict(input_prices):
    """
    input_prices: should be np size (batch_size, lookback_days) such as (6509, 7)

    """
    loaded_model = load_model()
    X_scaler, y_scaler = load_scaler()
    device = get_device()
    X = to_torch(np.expand_dims(X_scaler.transform(input_prices), -1))

    y_predicted = loaded_model(X.to(device)).detach().cpu().numpy().flatten()
    next_price_predicted = y_scaler.inverse_transform(y_predicted.reshape(-1, 1))

    predict_log_data = {
        "input_previous_n_price": X,
        "next_price_predicted": next_price_predicted,
    }
    logging.debug(f"Prediction: {predict_log_data}")
    return next_price_predicted


# def data(filepath):
#     # tbd
#     # return load_data(filepath)
#     return


# def retrain(tsize=0.1, model_name="model.joblib"):
#     """train or Retrains the model"""
#     #tbd

#     # df = data()
#     # y = df["Height"].values  # Target
#     # y = y.reshape(-1, 1)
#     # X = df["Weight"].values  # Feature(s)
#     # X = X.reshape(-1, 1)
#     # scaler = StandardScaler()
#     # X_scaler = scaler.fit(X)
#     # X = X_scaler.transform(X)
#     # y_scaler = scaler.fit(y)
#     # y = y_scaler.transform(y)
#     # X_train, X_test, y_train, y_test = train_test_split(
#     #     X, y, test_size=tsize, random_state=3
#     # )
#     # clf = Ridge()
#     # model = clf.fit(X_train, y_train)
#     # accuracy = model.score(X_test, y_test)
#     # logging.debug(f"Model Accuracy: {accuracy}")
#     # joblib.dump(model, model_name)
#     # return accuracy, model_name
#     return
