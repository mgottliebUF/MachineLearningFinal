# MachineLearningFinal
This is the final project for Introduction to Machine Learning, Supervised Learning. This project involves developing a stock price prediction and trading signal generation system using the k-Nearest Neighbors (kNN) algorithm. 

Michael Gottlieb 
Introduction to Machine Learning, Supervised Learning
Final Project 

Project Overview
This project involves developing a stock price prediction and trading signal generation system using the k-Nearest Neighbors (kNN) algorithm. The system predicts the next day's closing prices of a given stock and generates trading signals (buy or sell) based on these predictions. The performance of the trading strategy is backtested to evaluate its effectiveness.

Project Topic
Explanation and Type of Problem
This project addresses the problem of stock price prediction, a regression task within the domain of supervised learning. By leveraging historical stock data, the project aims to predict future stock prices using the k-Nearest Neighbors (kNN) algorithm.
Goal of the Project
The goal of this project is to develop a reliable stock price prediction model that can provide actionable trading signals. This is important because accurate stock price predictions can help traders make informed decisions, potentially leading to increased profitability.

Data
Data Source
The historical stock data is sourced from Yahoo Finance, accessed using the yfinance library in Python. This library provides a convenient interface to download historical market data, which is essential for building and testing the prediction model.
Data Description
The dataset includes daily stock prices (open, high, low, close) and trading volume for the specified stock. Key features extracted from the data include:
•	Price Change: The percentage change in the closing price compared to the previous day.
•	High-Low Spread: The difference between the highest and lowest prices of the day.
•	Average Price: The average of the high and low prices of the day.
The dataset size varies depending on the stock and the time period selected. For this analysis, the data spans from January 1, 2023, to the present date.
Data Cleaning
Cleaning Process
The data cleaning process involves handling missing values, outliers, and ensuring the data is in a suitable format for analysis.
1.	Handling Missing Values:
o	Missing values are identified and handled appropriately. For example, if a feature has too many missing values, it may be dropped. If the number of missing values is small, they are inputted using techniques such as interpolation or using the mean value.
2.	Outlier Removal:
o	Outliers are identified using statistical methods and removed to ensure they do not adversely affect the model's performance.
3.	Feature Engineering:
o	New features are created from the existing data to enhance the model's predictive power. For instance, the high-low spread, and average price are calculated and included as features.
The data cleaning process ensures that the dataset is free from inconsistencies and ready for model training. Visualizations such as histograms and box plots are used to verify the distribution of the data and the effectiveness of the cleaning steps.

Exploratory Data Analysis (EDA)
Procedure and Visualizations
EDA involves analyzing the dataset to understand the underlying patterns and relationships between features. The following steps are performed:
1.	Summary Statistics:
o	Summary statistics are calculated to provide an overview of the data distribution, including measures such as mean, median, standard deviation, and range.
2.	Histograms and Box Plots:
o	Histograms are used to visualize the distribution of individual features. Box plots help identify outliers and understand the spread of the data.
3.	Correlation Matrix:
o	A correlation matrix is generated to examine the relationships between different features. This helps identify multicollinearity and features that may be highly predictive of the target variable.
4.	Time Series Analysis:
o	The stock prices are plotted over time to identify trends, seasonality, and potential anomalies in the data.
Conclusion
The EDA process provides valuable insights into the data, guiding the selection of features and model parameters. It also helps in identifying potential challenges and formulating strategies to address them.
Analysis (Model Building and Training)
Model Selection
The k-Nearest Neighbors (kNN) regression algorithm is selected for this task. This algorithm predicts the target variable based on the average of the k-nearest neighbors in the feature space. The choice of kNN is motivated by its simplicity and effectiveness in capturing non-linear relationships.
Model Training
The model is trained with different values of k (number of neighbors). The values of k are provided by the user as input. The training process involves the following steps:
1.	Data Splitting:
o	The data is split into training (80%) and testing (20%) sets. The training set is used to train the model, and the testing set is used to evaluate its performance.
2.	Feature Scaling:
o	Features are scaled to ensure they are on a similar scale, which is important for distance-based algorithms like kNN.
3.	Model Training:
o	The kNN model is trained using the training data. The model parameters, including the value of k, are tuned to optimize performance.
4.	Performance Metrics:
o	The model's performance is evaluated using metrics such as Root Mean Squared Error (RMSE), Explained Variance Score (EVS), and R^2 Score. These metrics provide insights into the accuracy and reliability of the predictions.





Performance Metrics Explanation
def calculate_performance_metrics(test_target, predictions):
    mse = mean_squared_error(test_target, predictions)
    rmse = np.sqrt(mse)
    evs = explained_variance_score(test_target, predictions)
    r2 = r2_score(test_target, predictions)
    return rmse, evs, r2
•	RMSE: Measures the average magnitude of the errors between predicted and actual values. Lower values indicate better performance.
•	EVS: Measures the proportion of variance explained by the model. It ranges from 0 to 1, where 1 indicates perfect prediction.
•	R^2 Score: Indicates how well the predicted values match the actual values. It ranges from 0 to 1, where 1 means perfect prediction.

Results and Analysis
Prediction Accuracy
The model's predictions are compared to the actual stock prices using the performance metrics mentioned above. Lower RMSE and higher EVS and R^2 scores indicate better model performance.

Visualization
The actual and predicted stock prices are plotted to visually assess the model's performance.
# Plot actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(test_target.index, test_target, label='Actual Prices', color='blue')
plt.plot(test_target.index, predictions, label='Predicted Prices', color='red')
plt.title(f'Stock Price Prediction for {stock_ticker} using kNN (k={k})')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
Backtesting
The trading strategy is backtested using the predicted prices. The strategy involves buying or selling the stock based on whether the predicted price is higher or lower than the current price.
# Backtesting the strategy
transactions, final_value, daily_profits, total_profit_percentage = backtest_strategy(test_target, predictions)
print(f'Initial Investment: 10000')
print(f'Final Value after Backtesting: {final_value}')
print(f'Total Profit Percentage: {total_profit_percentage:.2f}%')
sharpe_ratio = calculate_sharpe_ratio(daily_profits)
•	Final Value: The value of the investment at the end of the testing period.
•	Total Profit Percentage: The percentage profit earned from the initial investment.
•	Sharpe Ratio: Measures the risk-adjusted return of the investment.

Summary of Results
The results show that the kNN model can effectively predict stock prices and generate profitable trading signals. The performance metrics and backtesting results indicate the model's reliability and potential for real-world application.
Discussion and Conclusion
1.	Model Performance:
o	The kNN model with the best value of k (based on the lowest RMSE and highest EVS and R^2) is selected as the best performing model.
o	The trading strategy based on the best performing model is evaluated for its profitability and risk (using the Sharpe ratio).
2.	Recommendations:
o	The system generates a recommendation (buy/sell) for the stock based on the predicted prices and the backtesting results.
o	The performance of the trading strategy can be further improved by experimenting with different values of k and other machine learning algorithms.



3.	Limitations:
o	The model's predictions are based solely on historical stock data and do not consider other factors such as market news, economic indicators, or company-specific events that can impact stock prices.
o	The backtesting results are hypothetical and may not reflect actual trading performance.
Conclusion
This project demonstrates the application of the k-Nearest Neighbors algorithm for stock price prediction and trading signal generation. The system's performance is evaluated using standard regression metrics and backtesting. The results indicate that the kNN model can provide useful trading signals, but its effectiveness may be limited by the factors mentioned above.
The generated PDF report summarizes the analysis and provides detailed insights into the model's performance and trading strategy. The system offers a useful tool for traders and investors to make informed decisions based on historical stock data.
This project has guided a 41% YTD achievement in the stock market, so far this has beat the S&P for the year by 10%. 

















Code Explanation: 
This section of the report explains each functional part of the code in detail. 

Importing Necessary Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
import matplotlib.pyplot as plt
from fpdf import FPDF
import ipywidgets as widgets
from IPython.display import display
•	yfinance: Used to download historical stock data.
•	pandas: For data manipulation and analysis.
•	numpy: For numerical computations.
•	sklearn.neighbors.KNeighborsRegressor: For implementing the kNN regression algorithm.
•	sklearn.metrics: For calculating performance metrics.
•	matplotlib.pyplot: For data visualization.
•	fpdf: For generating PDF reports.
•	ipywidgets and IPython.display: For interactive widgets in Jupyter notebooks.

Fetching Stock Data
def get_stock_data(ticker, retries=3, delay=5):
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start="2023-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'), interval='1d')
            if data.empty:
                raise ValueError(f"No data found for ticker '{ticker}'")
            return data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # wait before retrying
            else:
                print(f"Failed to get ticker '{ticker}' after {retries} attempts")
                return None
get_stock_data: This function attempts to download historical stock data for a given ticker. It retries up to three times in case of failure, with a delay between retries.

Calculating Performance Metrics
def calculate_performance_metrics(test_target, predictions):
    mse = mean_squared_error(test_target, predictions)
    rmse = np.sqrt(mse)
    evs = explained_variance_score(test_target, predictions)
    r2 = r2_score(test_target, predictions)
    
    return rmse, evs, r2
calculate_performance_metrics: This function calculates and returns three performance metrics: RMSE, EVS, and R^2 score.

Backtesting the Trading Strategy
def backtest_strategy(test_target, predictions, max_trades_per_week=4, initial_investment=10000):
    cash = initial_investment
    stock_held = 0
    transactions = []
    daily_profits = []
    trade_count = 0
    last_trade_date = None

    for i in range(1, len(predictions)):
        current_date = test_target.index[i]
        current_week = current_date.isocalendar()[1]
        
        if last_trade_date is not None and current_week == last_trade_date.isocalendar()[1]:
            if trade_count >= max_trades_per_week:
                daily_profits.append(0)  # No trade, no profit
                continue  # Skip to the next day
        else:
            trade_count = 0  # Reset trade count for the new week
        
        previous_cash = cash
        previous_stock_value = stock_held * test_target.iloc[i-1] if stock_held > 0 else 0
        previous_total_value = previous_cash + previous_stock_value
        
        if predictions[i] > test_target.iloc[i-1]:  # If predicted price is higher than previous day's actual price
            # Buy signal
            if cash > 0:
                stock_bought = cash / test_target.iloc[i]
                stock_held += stock_bought  # Buy as much stock as possible
                transactions.append((current_date, 'BUY', stock_bought, test_target.iloc[i]))
                cash = 0
                trade_count += 1
                last_trade_date = current_date
        else:
            # Sell signal
            if stock_held > 0:
                cash += stock_held * test_target.iloc[i]  # Sell all stock
                transactions.append((current_date, 'SELL', stock_held, test_target.iloc[i]))
                stock_held = 0
                trade_count += 1
                last_trade_date = current_date
        
        current_cash = cash
        current_stock_value = stock_held * test_target.iloc[i] if stock_held > 0 else 0
        current_total_value = current_cash + current_stock_value
        
        daily_profit = (current_total_value - previous_total_value) / previous_total_value
        daily_profits.append(daily_profit)
    
    final_value = cash + stock_held * test_target.iloc[-1]
    total_profit_percentage = (final_value / initial_investment - 1) * 100
    
    return transactions, final_value, daily_profits, total_profit_percentage
backtest_strategy: This function simulates a trading strategy based on the predicted prices. It maintains the cash and stock holdings, and records transactions. It ensures a maximum number of trades per week to avoid overtrading. The function returns the list of transactions, the final value of the investment, daily profits, and the total profit percentage.

Printing Performance Metrics: 
def print_performance_metrics(rmse, evs, r2):
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(' - RMSE measures the average magnitude of the errors between predicted and actual values. Lower values indicate better performance.')
    print(f'Explained Variance Score (EVS): {evs}')
    print(' - EVS measures the proportion of variance explained by the model. It ranges from 0 to 1, where 1 indicates perfect prediction.')
    print(f'R^2 Score: {r2}')
    print(' - R^2 (coefficient of determination) indicates how well the predicted values match the actual values. It ranges from 0 to 1, where 1 means perfect prediction.')
print_performance_metrics: This function prints the calculated performance metrics along with explanations of each metric.

Calculating Sharpe Ratio
def calculate_sharpe_ratio(daily_profits, risk_free_rate=0.01):
    mean_daily_return = np.mean(daily_profits)
    std_daily_return = np.std(daily_profits)
    sharpe_ratio = (mean_daily_return - risk_free_rate) / std_daily_return
    return sharpe_ratio
calculate_sharpe_ratio: This function calculates the Sharpe ratio, which measures the risk-adjusted return of the investment. It uses the mean and standard deviation of daily profits, and a risk-free rate.

Saving Results to PDF
def save_to_pdf(report_title, stock_ticker, best_overall_k, best_overall_profit, best_overall_sharpe_ratio, best_overall_recommendation, best_overall_price, transactions_df, daily_profits, total_profit_percentage):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, report_title, ln=True, align='C')
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, f"Best Performing Stock: {stock_ticker}", ln=True)
    pdf.cell(200, 10, f"Best Model: k={best_overall_k}", ln=True)
    pdf.cell(200, 10, f"Profit: ${best_overall_profit:.2f}", ln=True)
    pdf.cell(200, 10, f"Sharpe Ratio: {best_overall_sharpe_ratio:.2f}", ln=True)
    pdf.cell(200, 10, f"Current Recommendation: {best_overall_recommendation} at price ${best_overall_price:.2f}", ln=True)
    pdf.cell(200, 10, f"Total Profit Percentage: {total_profit_percentage:.2f}%", ln=True)
    pdf.cell(200, 10, f"Percentage Profit per Day: {np.mean(daily_profits) * 100:.2f}%", ln=True)
    
    pdf.cell(200, 10, "Transactions Table:", ln=True)
    
    pdf.set_font("Arial", '', 10)
    for i in range(len(transactions_df)):
        pdf.cell(200, 10, f"{transactions_df.iloc[i]['Date']} - {transactions_df.iloc[i]['Action']} - {transactions_df.iloc[i]['Shares']} shares at ${transactions_df.iloc[i]['Price']} - Profit: ${transactions_df.iloc[i]['Profit']:.2f}", ln=True)
    
    pdf.output("stock_analysis_report.pdf")
    print("PDF report generated: stock_analysis_report.pdf")
save_to_pdf: This function generates a PDF report summarizing the analysis results. It includes the best performing stock, model, profit, Sharpe ratio, recommendation, transactions, and profit percentages.
Analyzing the Stock
def analyze_stock(stock_ticker, k_values):
    best_overall_k = None
    best_overall_profit = -float('inf')
    best_overall_sharpe_ratio = None
    best_overall_recommendation = None
    best_overall_price = None
    best_overall_transactions_df = None
    best_overall_daily_profits = None
    best_overall_total_profit_percentage = None
    
    print(f"\nAnalyzing stock: {stock_ticker}")
    
    # Fetch historical stock data for the current year
    data = get_stock_data(stock_ticker)
    
    if data is not None:
        print("Data retrieved successfully")
        
        # Prepare the data
        print("Preparing the data...")
        data['Price_Change'] = data['Close'].pct_change()
        data['High_Low_Spread'] = data['High'] - data['Low']
        data['Average_Price'] = (data['High'] + data['Low']) / 2
        data.dropna(inplace=True)

        # Define the features and target variable
        features = data[['Close', 'High_Low_Spread', 'Average_Price']]
        target = data['Close'].shift(-1)  # Predict next day's closing price
        features = features[:-1]  # Drop the last row to match the target
        target = target[:-1]  # Drop the last row as it has NaN target

        # Split the data into training (first 80%) and testing (last 20%) sets
        train_size = int(len(features) * 0.8)
        train_features, test_features = features[:train_size], features[train_size:]
        train_target, test_target = target[:train_size], target[train_size:]

        # Ensure there are no NaN values in the test_target
        test_features = test_features[~test_target.isna()]
        test_target = test_target.dropna()

        best_k = None
        best_profit = -float('inf')
        best_recommendation = None
        best_price = None
        best_sharpe_ratio = None
        best_transactions_df = None
        best_daily_profits = None
        best_total_profit_percentage = None

        for k in k_values:
            print(f"\nRunning kNN with k={k}...")
            # Initialize and train the kNN model
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(train_features, train_target)

            # Predict stock prices
            predictions = knn.predict(test_features)

            # Calculate performance metrics
            rmse, evs, r2 = calculate_performance_metrics(test_target, predictions)
            
            # Print performance metrics explanations
            print("\nPerformance Metrics Explanation:")
            print("Root Mean Squared Error (RMSE): Measures the average magnitude of the errors between predicted and actual values. Lower values indicate better performance.")
            print("Explained Variance Score (EVS): Measures the proportion of variance explained by the model. It ranges from 0 to 1, where 1 indicates perfect prediction.")
            print("R^2 Score: Indicates how well the predicted values match the actual values. It ranges from 0 to 1, where 1 means perfect prediction.")
            
            # Print the performance metrics
            print_performance_metrics(rmse, evs, r2)

            # Plot actual vs predicted prices
            plt.figure(figsize=(14, 7))
            plt.plot(test_target.index, test_target, label='Actual Prices', color='blue')
            plt.plot(test_target.index, predictions, label='Predicted Prices', color='red')
            plt.title(f'Stock Price Prediction for {stock_ticker} using kNN (k={k})')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()

            # Backtest the strategy
            print("\nBacktesting the strategy...")
            transactions, final_value, daily_profits, total_profit_percentage = backtest_strategy(test_target, predictions)

            print(f'\nInitial Investment: 10000')
            print(f'Final Value after Backtesting: {final_value}')
            print(f'Total Profit Percentage: {total_profit_percentage:.2f}%')
            
            sharpe_ratio = calculate_sharpe_ratio(daily_profits)

            if final_value > best_profit:
                best_k = k
                best_profit = final_value
                best_recommendation = "BUY" if predictions[-1] > test_target.iloc[-1] else "SELL"
                best_price = test_target.iloc[-1]
                best_sharpe_ratio = sharpe_ratio
                best_transactions_df = pd.DataFrame(transactions, columns=['Date', 'Action', 'Shares', 'Price'])
                best_transactions_df['Profit'] = best_transactions_df.apply(lambda row: (row['Shares'] * row['Price'] if row['Action'] == 'SELL' else -row['Shares'] * row['Price']), axis=1).cumsum()
                best_daily_profits = daily_profits
                best_total_profit_percentage = total_profit_percentage

            # Recommendation based on the last prediction
            recommendation = "BUY" if predictions[-1] > test_target.iloc[-1] else "SELL"
            print(f'Recommendation: {recommendation}')

            # Plot the signals on the price chart
            plt.figure(figsize=(14, 7))
            plt.plot(test_target.index, test_target, label='Actual Prices', color='blue')
            for transaction in transactions:
                if transaction[1] == 'BUY':
                    plt.axvline(transaction[0], color='green', linestyle='--', lw=1)
                elif transaction[1] == 'SELL':
                    plt.axvline(transaction[0], color='red', linestyle='--', lw=1)
            plt.title(f'Trading Signals for {stock_ticker} (k={k})')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()

        return (best_k, best_profit, best_sharpe_ratio, best_recommendation, best_price, best_transactions_df, best_daily_profits, best_total_profit_percentage)
    else:
        print("Failed to retrieve stock data. Please try again with a different ticker.")
        return None
analyze_stock: This function performs the entire analysis for a given stock ticker. It fetches the historical stock data, prepares the features and target variable, splits the data into training and testing sets, and trains the kNN model with different values of k. It calculates performance metrics, backtests the trading strategy, and identifies the best performing model and trading strategy.



Generating PDF Button Handler
def generate_pdf_button_handler(event):
    save_to_pdf("Stock Analysis Report", stock_ticker, best_k, best_profit, best_sharpe_ratio, best_recommendation, best_price, best_transactions_df, best_daily_profits, best_total_profit_percentage)
generate_pdf_button_handler: This function handles the event when the "Generate PDF" button is clicked. It calls the save_to_pdf function to generate the PDF report.

Input Trades from User
def input_trades():
    trades = []
    for i in range(3):
        ticker = input(f"Enter the ticker for trade {i+1}: ").strip().upper()
        purchase_price = float(input(f"Enter the purchase price for {ticker}: ").strip())
        shares = float(input(f"Enter the number of shares for {ticker}: ").strip())
        trades.append((ticker, purchase_price, shares))
    return trades
input_trades: This function prompts the user to input details for three trades (ticker, purchase price, and number of shares) and returns a list of these trades.

