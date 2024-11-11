import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objs as go

# Title of the app
st.title('📈 Stock Price Prediction Web App')

# Add logo to the sidebar
st.sidebar.image('Designer.jpeg', width=100)  # Adjust the width as needed

# Sidebar for user input
st.sidebar.header('Input Stock Parameters')

# Select box for stock symbol with a few options
stock_symbol = st.sidebar.selectbox('Select Stock Symbol:', ['AAPL', 'NFLX', 'GOOGL', 'AMZN', 'MSFT'], index=0)

# Slider for date range selection
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))

# Fetch stock data from Yahoo Finance
st.write(f"Fetching stock data for **{stock_symbol}** from **{start_date}** to **{end_date}**...")
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Check if data is valid
if data.empty or len(data) < 2:
    st.error(f"Error: No or insufficient data found for {stock_symbol}. Please enter a valid stock symbol.")
else:
    # Show raw data with a nice header
    st.subheader(f'📊 {stock_symbol} Stock Data')
    st.write(data.tail())

    # Prepare data for Prophet model
    data.reset_index(inplace=True)
    data = data[['Date', 'Close']]
    data.columns = ['ds', 'y']  # Prophet expects 'ds' and 'y' column names

    # Train Prophet model
    model = Prophet()
    model.fit(data)

    # Make future predictions
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Show forecast results
    st.subheader(f'📈 Forecasting {stock_symbol} Stock Price')
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Convert 'ds' to datetime and 'yhat' to numeric
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    forecast['yhat'] = pd.to_numeric(forecast['yhat'], errors='coerce')

    # Handle invalid data
    forecast.dropna(subset=['ds', 'yhat'], inplace=True)

    # Line chart for predictions
    st.line_chart(forecast[['ds', 'yhat']].set_index('ds'))

    # Plot forecast components
    st.subheader(f'{stock_symbol} Stock Forecast Components')
    fig = model.plot_components(forecast)
    st.write(fig)

# Add footer
st.markdown('---')
st.write("Made with ❤️ by Ujjwal")