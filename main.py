import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from model import prepare_data, build_model

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        return stock_data
    if 'Date' not in stock_data.columns:
        stock_data.reset_index(inplace=True)  # Ensure Date is in a column
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])  # Convert Date column to datetime
    return stock_data

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Input(id='stock-ticker', value='', type='text'),
    dcc.DatePickerRange(
        id='date-picker',
        start_date='2023-01-01',
        end_date='2024-01-01',
    ),
    html.Button('Submit', id='submit-button', n_clicks=0),
    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            dcc.Graph(id='stock-graph'),
        ]
    ),
    dcc.Loading(
        id="loading-2",
        type="circle",
        children=[
            dcc.Graph(id='predicted-graph'),
        ]
    )
])

@app.callback(
    Output('stock-graph', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('stock-ticker', 'value'),
     dash.dependencies.State('date-picker', 'start_date'),
     dash.dependencies.State('date-picker', 'end_date')]
)
def update_actual_graph(n_clicks, ticker, start_date, end_date):
    if (ticker==""):
      return {
                  'data': [],
                  'layout': go.Layout(title='No data available for the selected range')
              }

    df = fetch_stock_data(ticker, start_date, end_date)

    if df.empty:
        return {
            'data': [],
            'layout': go.Layout(title='No data available for the selected range')
        }

    trace = go.Scatter(
        x=df['Date'],  # Ensure Date column is used
        y=df['Close'],
        mode='lines',
        name='Actual'
    )
    
    return {
        'data': [trace],
        'layout': go.Layout(title='Actual Stock Prices', xaxis_title='Date', yaxis_title='Close Price')
    }

@app.callback(
    Output('predicted-graph', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('stock-ticker', 'value'),
     dash.dependencies.State('date-picker', 'start_date'),
     dash.dependencies.State('date-picker', 'end_date')]
)
def update_predicted_graph(n_clicks, ticker, start_date, end_date):
    if (ticker==""):
      return {
                  'data': [],
                  'layout': go.Layout(title='No data available for the selected range')
              }
    df = fetch_stock_data(ticker, start_date, end_date)

    if df.empty:
        return {
            'data': [],
            'layout': go.Layout(title='No data available for the selected range')
        }

    x_data, y_data, scaler = prepare_data(df)

    model = build_model((x_data.shape[1], 1))
    model.fit(x_data, y_data, epochs=100, batch_size=32)
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    total_days = (end_date - start_date).days
    
    look_back = len(df)
    
    last_days = df[['Close']].tail(look_back).values
    last_days_scaled = scaler.transform(last_days)
    
    future_days = int(total_days * 0.1)

    predictions = []
    current_batch = np.reshape(last_days_scaled, (1, look_back, 1))

    for _ in range(future_days):
        pred = model.predict(current_batch)
        predictions.append(pred[0, 0])
        pred = np.reshape(pred, (1, 1, 1))  # Ensure pred has shape (1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], pred, axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
    trace = go.Scatter(
        x=future_dates,
        y=predictions.flatten(),
        mode='lines',
        name='Predicted'
    )
    
    return {
        'data': [trace],
        'layout': go.Layout(title='Predicted Stock Prices', xaxis_title='Date', yaxis_title='Close Price')
    }

if __name__ == '__main__':
    app.run_server(debug=True)
