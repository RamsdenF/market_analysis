import pandas as pd
import matplotlib.pyplot as plt
import load_data

# Load and prepare data
csv_folder = 'Market_Data_2013'
data_frames = load_data.load_data(csv_folder)

# Combine data into one DataFrame and convert to numeric
prices = pd.DataFrame()
for file_name, df in data_frames.items():
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce').ffill()
    prices[file_name.replace('.csv', '')] = df['PRICE']

# Additional check: Forward-fill any NaN values in the combined DataFrame
prices.ffill(inplace=True)

# Calculate daily returns
returns = prices.pct_change().dropna()

# Define the event date and window
event_date = pd.Timestamp('2020-03-11')
pre_event_window = 90
post_event_window = 90

# Calculate average returns before and after the event
avg_pre_event_returns = returns.loc[event_date - pd.Timedelta(days=pre_event_window):event_date].mean()
avg_post_event_returns = returns.loc[event_date:event_date + pd.Timedelta(days=post_event_window)].mean()

# Plotting
plt.figure(figsize=(12, 6))
avg_pre_event_returns.plot(kind='bar', color='blue', position=0, width=0.4, label='Pre-Event')
avg_post_event_returns.plot(kind='bar', color='red', position=1, width=0.4, label='Post-Event')
plt.title('Average Daily Returns of Indices Before and After Covid-19 Declaration')
plt.xlabel('Index')
plt.ylabel('Average Daily Return')
plt.legend()
plt.grid(True)
plt.show()
