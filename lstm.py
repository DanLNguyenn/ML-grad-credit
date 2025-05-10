import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os

# device path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# getting user input based on stock selection and prediction preriod
valid_tickers = ["^GSPC", "^IXIC", "^FTSE", "^NSEI", "^BSESN", "^N225", "000001.SS", "^N100", "^DJI", "GC=F", "CL=F"]
ticker = input(f"Enter ticker to predict ({', '.join(valid_tickers)}): ").strip()
while ticker not in valid_tickers:
    ticker = input("Invalid ticker. Please enter a valid one: ").strip()

days_to_predict = input("Predict next 5 or 10 days? ").strip()
while days_to_predict not in ["5", "10"]:
    days_to_predict = input("Invalid input. Enter 5 or 10: ").strip()
PREDICT_DAYS = int(days_to_predict)

# load and concatenate all yearly files 
data_dir = "dataset"
all_years = []
for year in range(2008, 2024):
    filename = f"{year}_Global_Markets_Data.csv"
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    all_years.append(df)

# combine and sort data
stock_all = pd.concat(all_years).sort_values("Date").reset_index(drop=True)

stock_data = stock_all[stock_all["Ticker"] == ticker].sort_values("Date")

# get last 1000 days of the selected stock
close_prices = stock_data["Close"].values[-1000:].reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)

# hyperparameters tuning
INPUT_WINDOW = 60
EPOCHS = 20
BATCH_SIZE = 32

# dataset 
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

dataset = TimeSeriesDataset(scaled_data, INPUT_WINDOW, PREDICT_DAYS)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# model
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=PREDICT_DAYS):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out)

model = LSTMForecast().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training model
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# predict
model.eval()
last_window = torch.FloatTensor(scaled_data[-INPUT_WINDOW:]).unsqueeze(0).to(device)
with torch.no_grad():
    pred_scaled = model(last_window).cpu().numpy()

# inverse scaling
pred_prices = scaler.inverse_transform(pred_scaled.reshape(-1, 1))

# output
print(f"\nNext {PREDICT_DAYS} day predicted prices for {ticker}:")
for i, price in enumerate(pred_prices.flatten(), 1):
    print(f"Day {i}: ${price:.2f}")

# prediction plot
plt.figure(figsize=(10, 5))
plt.plot(range(INPUT_WINDOW), scaler.inverse_transform(scaled_data[-INPUT_WINDOW:]), label="Last 60 Days")
plt.plot(range(INPUT_WINDOW, INPUT_WINDOW + PREDICT_DAYS), pred_prices, marker='o', label=f"Next {PREDICT_DAYS} Days")
plt.title(f"{ticker} Forecast")
plt.xlabel("Time (Days)")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
