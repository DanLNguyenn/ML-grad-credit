How to compile and run the program:
 - Compile:
  + Run all cells in the eda.ipynb
  + In the terminal, run python lstm.py

- Running the program:
 + When the program started, user will be prompted to enter a stock based on the list of tickers given (such as ^GSPC, etc.)
 + After that, user will be prompted to enter either a 5 or 10 days prediction

Program output:
 - The program will output 2 graphs representing the result of the 200 backtesting data and the model prediction based on the past 60 days window and all of the dataset training. 

Model description:
The program utilized the model LSTM implemented in python to perform time series prediction on a stock market dataset. In details, the model predicts the next 5 or 10 days of closing prices for the selected stock from the list. This is based on the sliding window of the previous 60 days.

Additionally, the model is trained based on the last 1000 days of the selected ticker and tested on the 200 days before those. The Close price is extracted, scaled to the range [0, 1] using MinMaxScaler, and then split into overlapping sequences of 60 input days to predict the following 5 or 10 output days.

Midek architecture:
 - Input size: 1 (Closing price only)
 - Sequence length: 60 time steps
 - LSTM layers: 2 stacked layers
 - Hidden size: 64 units
 - Output layer: Fully connected layer producing either 5 or 10 output values (one per day)
 - Loss function: Root Mean Squared Error (RMSE)
 - Optimizer: Adam, with a learning rate of 0.001

Training process:
 - The model is trained over 20 epochs using a batch size of 32.
 - For each epoch, the model minimizes the MSE loss between predicted and actual closing prices.
 - Trained exclusively on the most recent 1000 data points for maximum relevance and avoid overfitting to too much past noises or trends.
 - After the training process, the model is tested on the past 200 datatpoints before the 1000 training data and a graph is produced to demonstrate the performance.
 - After the testing, the model is going to predict based on the selected 5 or 10 days.

Evaluation and Accuracy:
 - Model performance is evaluated on the 200 days before the training data using a sliding window approach.
 - Evaluation uses only the first predicted day from each output window for error computation.
 - Test RMSE is printed in the terminal.
 - Actual vs. predicted prices are plotted for visualization.






    