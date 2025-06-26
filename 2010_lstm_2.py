import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset

INPUT_WINDOW = 60
EPOCHS = 20
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
PREDICT_DAYS = int(input("Predict next 5 or 10 days? (5/10): ").strip())
while PREDICT_DAYS not in [5, 10]:
    PREDICT_DAYS = int(input("Invalid input. Enter 5 or 10: ").strip())

# directories
train_dir = "test_dataset_2/train"
test_dir = "test_dataset_2/test"
plot_dir = "test_dataset_2/plots"
os.makedirs(plot_dir, exist_ok=True)

# dataset class
class TimeSeriesDataset(torch.utils.data.Dataset):
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

# RMSE
rmses = {}
for filename in os.listdir(train_dir):
    if not filename.endswith(".csv"):
        continue

    ticker = filename.replace(".csv", "")
    train_path = os.path.join(train_dir, filename)
    test_path = os.path.join(test_dir, filename)

    if not os.path.exists(test_path):
        print(f"⚠️ Skipping {ticker}: test file missing.")
        continue

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if len(train_df) < INPUT_WINDOW or len(test_df) < PREDICT_DAYS:
        print(f"⚠️ Skipping {ticker}: insufficient data.")
        continue

    # scaler
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_df["Close"].values.reshape(-1, 1))

    # prepare datas
    dataset = TimeSeriesDataset(scaled_train, INPUT_WINDOW, PREDICT_DAYS)
    if len(dataset) == 0:
        print(f"⚠️ Skipping {ticker}: empty dataset after windowing.")
        continue

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # model
    model = LSTMForecast().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training
    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # predict based on select day
    model.eval()
    with torch.no_grad():
        last_window = torch.FloatTensor(scaled_train[-INPUT_WINDOW:]).unsqueeze(0).to(device)
        pred_scaled = model(last_window).cpu().numpy()

    pred_prices = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    actual_raw = test_df["Close"].values[:PREDICT_DAYS].reshape(-1, 1)
    scaled_actual = scaler.transform(actual_raw)
    actual_prices = scaler.inverse_transform(scaled_actual)

    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    rmses[ticker] = rmse
    print(f"✅ {ticker}: RMSE = {rmse:.2f}")

# plot
tickers = list(rmses.keys())
rmse_values = [rmses[t] for t in tickers]

plt.figure(figsize=(12, 6))
bars = plt.bar(tickers, rmse_values, color='skyblue')
plt.xticks(rotation=45)
plt.ylabel("RMSE")
plt.title(f"LSTM Forecast RMSE (Next {PREDICT_DAYS} Days)")
plt.tight_layout()

# RSME bar graph
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points", ha='center', va='bottom')

plot_path = os.path.join(plot_dir, "rmse_comparison.png")
plt.savefig(plot_path)
plt.close()
print(f"\n📊 RMSE comparison plot saved to: {plot_path}")
