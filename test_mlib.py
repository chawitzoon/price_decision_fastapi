from mlib.mlib import load_scaler, load_model, load_hyper_param, get_device, predict
from mlib.mlib_util import Agent, load_data, preprocess_data, to_torch, to_loader
import numpy as np

# filepath = 'D:/Project/ml_devops/aws_app_demo/mlib/AMZN.csv'
# lookback_days = 7

device = get_device()
# print(device)
# data = load_data(filepath)
# X_train, X_test, y_train, y_test, X_scaler, y_scaler = preprocess_data(data, lookback_days=lookback_days, train_test_split=0.9)

# X_train = to_torch(X_train)
# y_train = to_torch(y_train)
# X_test = to_torch(X_test)
# y_test = to_torch(y_test)

# train_loader = to_loader(X_train, y_train, batch_size = 16, train=True)
# test_loader = to_loader(X_test, y_test, batch_size = 16, train=False)

# Construct the full path to the JSON file
hyper_param = load_hyper_param()
lstm_agent = Agent(
    hyper_param["input_size"],
    hyper_param["hidden_size"],
    hyper_param["num_stacked_layers"],
    device=device,
)

# lstm_agent.train_loader = train_loader
# lstm_agent.test_loader = test_loader
# lstm_agent.X_scaler = X_scaler
# lstm_agent.y_scaler = y_scaler

# lstm_agent.train()
# lstm_agent.save_model_weight('mlib_model/lstm_model.pth')
# lstm_agent.save_hyper_param('mlib_model/lstm_model_hyper_param.json')

# Load the model
loaded_model = load_model()
lstm_agent.model = loaded_model

# lstm_agent.visualize(X_train, y_train)
# y_test = lstm_agent.predict(X_test.reshape((-1, lookback_days)))
# lstm_agent.visualize(X_test, y_test)

input_prices = np.array([[120, 121.4, 126.9, 128.0, 127.8, 129.1, 130.1]])
next_price_predicted = predict(input_prices)
print(next_price_predicted)
