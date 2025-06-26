import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error

INPUT_WINDOW = 60
EPOCHS = 20
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# user input
valid_tickers = ["^GSPC", "^IXIC", "^FTSE", "^NSEI", "^BSESN", "^N225", "000001.SS", "^N100", "^DJI", "GC=F", "CL=F"]
PREDICT_DAYS = int(input("Predict next 5 or 10 days? ").strip())

# set up directory
train_dir = "test_dataset_2/train"
test_dir = "test_dataset_2/test"
plot_dir = "test_dataset_2/plots"
os.makedirs(plot_dir, exist_ok=True)

# dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window, pred_days):
        self.X, self.y = [], []
        for i in range(input_window, len(data) - pred_days + 1):
            self.X.append(data[i - input_window:i])
            self.y.append(data[i:i + pred_days])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx]).squeeze()

# model
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=PREDICT_DAYS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


for ticker in valid_tickers:
    safe_name = ticker.replace("/", "_").replace("=", "")
    train_path = os.path.join(train_dir, f"{safe_name}.csv")
    test_path = os.path.join(test_dir, f"{safe_name}.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"{ticker}: missing files, skipping.")
        continue

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if len(train_df) < INPUT_WINDOW or len(test_df) < PREDICT_DAYS:
        print(f"{ticker}: insufficient data, skipping.")
        continue

    # scaler
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_df["Close"].values.reshape(-1, 1))

    train_dataset = TimeSeriesDataset(scaled_train, INPUT_WINDOW, PREDICT_DAYS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # set up model
    model = LSTMForecast().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    # prediction for the next 5 or 10 days
    model.eval()
    last_window = torch.FloatTensor(scaled_train[-INPUT_WINDOW:]).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_scaled = model(last_window).cpu().numpy()
    pred_prices = scaler.inverse_transform(pred_scaled.reshape(-1, 1))

    # scale actual price
    actual_raw = test_df["Close"].values[:PREDICT_DAYS].reshape(-1, 1)
    actual_prices = scaler.inverse_transform(scaler.transform(actual_raw))

    # RMSE
    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    print(f"{ticker}: RMSE = {rmse:.2f}")

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(PREDICT_DAYS), actual_prices, label="Actual", linewidth=2)
    plt.plot(range(PREDICT_DAYS), pred_prices, label="Predicted", linestyle="--", linewidth=2)
    plt.title(f"{ticker} - LSTM Forecast ({PREDICT_DAYS} Days)")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"{safe_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")
