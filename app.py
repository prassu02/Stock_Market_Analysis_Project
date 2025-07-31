import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

st.set_page_config(page_title="Apple Stock Forecast", layout="wide")

model = joblib.load('stock_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📈 Apple Stock Price Prediction - Next 30 Days")
st.markdown("Upload your Apple stock CSV file with columns: `Date, Open, High, Low, Close, Volume`")

uploaded_file = st.file_uploader("cleaned _data.csv", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    st.subheader("📋 Last 5 Rows of Uploaded Data")
    st.write(df.tail())

    last_30 = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(30)
    input_scaled = scaler.transform(last_30)
    X_input = input_scaled.reshape(1, -1)

    predictions_scaled = []
    for _ in range(30):
        pred = model.predict(X_input)[0]
        predictions_scaled.append(pred)

        next_day = np.copy(X_input).reshape(30, 5)
        next_row = next_day[1:]
        next_row = np.vstack([next_row, [0, 0, 0, pred, 0]])
        X_input = next_row.reshape(1, -1)

    preds = np.array(predictions_scaled).reshape(-1, 1)
    dummy = np.zeros((30, 5))
    dummy[:, 3] = preds[:, 0]
    inv_preds = scaler.inverse_transform(dummy)[:, 3]

    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(30)]
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': inv_preds})

    st.subheader("📈 Forecasted Closing Prices")
    st.write(forecast_df)

    st.line_chart(forecast_df.set_index('Date'))
