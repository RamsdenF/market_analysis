import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_data

# Load and prepare data
csv_folder = 'Market_Data_2013'
data_frames = load_data.load_data(csv_folder)

# Combine data into one DataFrame and convert all values to numeric
prices = pd.DataFrame()
for file_name, df in data_frames.items():
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    # Convert 'PRICE' to numeric, forward-fill NaN values
    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce').ffill()
    prices[file_name.replace('.csv', '')] = df['PRICE']

# Ensure no NaN values remain
prices.ffill(inplace=True)

# Calculate daily returns
returns = prices.pct_change().dropna()

# Mean returns and covariance
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Number of portfolios to simulate
num_portfolios = 10000

# Set up array to hold results, including weights
num_assets = len(prices.columns)
results = np.zeros((3 + num_assets, num_portfolios))

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_stddev

    results[0,i] = portfolio_return
    results[1,i] = portfolio_stddev
    results[2,i] = sharpe_ratio

    # Store the weights
    for j in range(len(weights)):
        results[j+3, i] = weights[j]

# Convert results array to DataFrame
columns = ['Return', 'StdDev', 'Sharpe'] + list(prices.columns)
portfolio_results = pd.DataFrame(results.T, columns=columns)

# Extract the portfolio with the highest Sharpe Ratio
best_sharpe_portfolio = portfolio_results.iloc[portfolio_results['Sharpe'].idxmax()]

# Extract the portfolio with the minimum standard deviation
min_vol_portfolio = portfolio_results.iloc[portfolio_results['StdDev'].idxmin()]

# Plotting the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_results.StdDev, portfolio_results.Return, c=portfolio_results.Sharpe, cmap='YlGnBu')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(best_sharpe_portfolio.iloc[1], best_sharpe_portfolio.iloc[0], marker='*', color='r', s=100, label='Maximum Sharpe ratio')
plt.scatter(min_vol_portfolio.iloc[1], min_vol_portfolio.iloc[0], marker='*', color='g', s=100, label='Minimum volatility')
plt.title('Efficient Frontier with Markowitz Portfolio Optimization')
plt.legend(labelspacing=0.8)
plt.show()

# Extract the portfolio with the highest Sharpe Ratio
best_sharpe_portfolio = portfolio_results.iloc[portfolio_results['Sharpe'].idxmax()]

# Print the asset allocation for the maximum Sharpe ratio portfolio
print("Asset Allocation for the Portfolio with Maximum Sharpe Ratio:")
for i, col in enumerate(prices.columns):
    weight = best_sharpe_portfolio[col]  # Accessing weight by column name
    print(f"{col}: {weight*100:.2f}%")

