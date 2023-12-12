import dash
from dash import dcc, html  # Updated imports
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import load_data

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and prepare data
csv_folder = 'Market_Data_2013'
data_frames = load_data.load_data(csv_folder)

# Ensure each DataFrame in data_frames has 'DATE' and 'PRICE' columns
# For example, if not already done:
# for df in data_frames.values():
#     df['DATE'] = pd.to_datetime(df['DATE'])
#     df.set_index('DATE', inplace=True)
#     df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce').ffill()

# App layout
app.layout = html.Div([
    html.H1("Stock Market Indices Dashboard"),
    dcc.Dropdown(
        id='index-dropdown',
        options=[{'label': k, 'value': k} for k in data_frames.keys()],
        value=list(data_frames.keys())[0]  # Set default value to the first index
    ),
    dcc.Graph(id='price-graph')
])

# Callback to update graph
@app.callback(
    Output('price-graph', 'figure'),
    [Input('index-dropdown', 'value')]
)
def update_graph(selected_index):
    df = data_frames[selected_index]
    df['DATE'] = pd.to_datetime(df['DATE'])

    # Convert 'PRICE' to numeric, replacing non-numeric values (like periods) with NaN
    # Then forward-fill any NaN values
    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce').ffill()

    # Calculate the rolling average
    df['ROLLING_AVG'] = df['PRICE'].rolling(window=30).mean()

    # Plot the rolling average
    fig = px.line(df, x='DATE', y='ROLLING_AVG', title=f'{selected_index} 30-Day Moving Average')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
