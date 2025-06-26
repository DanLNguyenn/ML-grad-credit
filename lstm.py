import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sser input
valid_tickers = ["^GSPC", "^IXIC", "^FTSE", "^NSEI", "^BSESN", "^N225", "000001.SS", "^N100", "^DJI", "GC=F", "CL=F"]
ticker = input(f"Enter ticker to predict ({', '.join(valid_tickers)}): ").strip()
while ticker not in valid_tickers:
    ticker = input("Invalid ticker. Please enter a valid one: ").strip()

days_to_predict = input("Predict next 5 or 10 days? ").strip()
while days_to_predict not in ["5", "10"]:
    days_to_predict = input("Invalid input. Enter 5 or 10: ").strip()
PREDICT_DAYS = int(days_to_predict)

# loading data and concatenating all yearly files
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

# split data into training and testing
test_data = stock_data["Close"].values[-1200:-1000].reshape(-1, 1)  # 200 test points before the 1000 training points
train_data = stock_data["Close"].values[-1000:].reshape(-1, 1)      # 1000 points for training

# scale both using scaler fitted on training data
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)

# hyperparameters tunning
INPUT_WINDOW = 60
EPOCHS = 20
BATCH_SIZE = 32

# using dataset 
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

dataset = TimeSeriesDataset(scaled_train, INPUT_WINDOW, PREDICT_DAYS)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# getting model
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=PREDICT_DAYS):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  
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

# evaluation on test set
model.eval()
X_test, y_test, y_pred = [], [], []

with torch.no_grad():
    for i in range(INPUT_WINDOW, len(scaled_test) - PREDICT_DAYS + 1):
        input_seq = torch.FloatTensor(scaled_test[i - INPUT_WINDOW:i]).unsqueeze(0).to(device)
        target_seq = scaled_test[i:i + PREDICT_DAYS]
        prediction = model(input_seq).cpu().numpy().reshape(-1, 1)
        
        # save for evaluation
        y_pred.append(prediction)
        y_test.append(target_seq)

# inverse transforming
y_pred = scaler.inverse_transform(np.concatenate(y_pred))
y_test = scaler.inverse_transform(np.concatenate(y_test))

# calculating RMSE 
rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
print(f"\nTest RMSE on past 200 points: {rmse:.2f}")

# prediction
model.eval()
last_window = torch.FloatTensor(scaled_train[-INPUT_WINDOW:]).unsqueeze(0).to(device)
with torch.no_grad():
    pred_scaled = model(last_window).cpu().numpy()

# inverse scaling
pred_prices = scaler.inverse_transform(pred_scaled.reshape(-1, 1))

# print to terminal
print(f"\nNext {PREDICT_DAYS} day predicted prices for {ticker}:")
for i, price in enumerate(pred_prices.flatten(), 1):
    print(f"Day {i}: ${price:.2f}")

# align true and predicted values 
true_vals = []
pred_vals = []

# compare the first predicted day for each sequence
for true_seq, pred_seq in zip(y_test, y_pred):
    true_vals.append(true_seq[0])     # actual next-day value
    pred_vals.append(pred_seq[0])     # predicted next-day value

# ploting the testing result
plt.figure(figsize=(12, 6))
plt.plot(true_vals, label="Actual Price", linewidth=2)
plt.plot(pred_vals, label="Predicted Price", linewidth=2, linestyle='--')
plt.title(f"{ticker} | LSTM Forecast - Test Set (Previous 200 Days)")
plt.xlabel("Time (Sliding Window Steps)")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ploting the prediction
plt.figure(figsize=(10, 5))
plt.plot(range(INPUT_WINDOW), scaler.inverse_transform(scaled_train[-INPUT_WINDOW:]), label="Last 60 Days")
plt.plot(range(INPUT_WINDOW, INPUT_WINDOW + PREDICT_DAYS), pred_prices, marker='o', label=f"Next {PREDICT_DAYS} Days")
plt.title(f"{ticker} Forecast")
plt.xlabel("Time (Days)")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
