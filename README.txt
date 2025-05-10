How to compile and run the program:
 - Compile:
  + Run all cells in the eda.ipynb
  + In the terminal, run python lstm.py

- Running the program:
 + When the program started, user will be prompted to enter a stock based on the list of tickers given (such as ^GSPC, etc.)
 + After that, user will be prompted to enter either a 5 or 10 days prediction

Program output:
 - The program will output a graph representing the model prediction based on the past 60 days window and all of the dataset training 

Model description:
The program utilized the model LSTM implemented in python to perform time series prediction on a stock market dataset. In details,
the model predicts the next 5 or 10 days of closing prices for the selected stock from the list. This is based on the sliding window of the
previous 60 days.

Additionally, the model is trained based on the last 1000 days of the selected ticker. The Close price is extracted, scaled to the 
range [0, 1] using MinMaxScaler, and then split into overlapping sequences of 60 input days to predict the following 5 or 10 output days.

Midek architecture:
 - Input size: 1 (only using closing price as input)
 - Sequence length: 60 time steps
 - LSTM layers: 2 stacked layers
 - Hidden size: 64 units
 - Output layer: Fully connected layer producing either 5 or 10 output values (one per day)
 - Loss function: Mean Squared Error (MSE)
 - Optimizer: Adam, with a learning rate of 0.001

Training process:
 - The model is trained over 20 epochs using a batch size of 32.
 - For each epoch, the model minimizes the MSE loss between predicted and actual closing prices.
 - At the end of training, the model is evaluated on the last 60-day window to forecast the next 5 or 10 days.

Accuracy:
 - This implementation does not include formal validation or test set evaluation (e.g., RMSE/MAE on unseen data), as it's focused on 
 forecasting from the most recent trend window.
 - There are also limitation on the GPU capacity so future improvement could be implementation of rolling window backtesting





    