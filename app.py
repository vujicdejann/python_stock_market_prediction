from flask import Flask, render_template, request
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import joblib

app = Flask(__name__)

# Učitavanje modela (model_random_forest.pkl)
model = joblib.load('model_random_forest.pkl')

def get_stock_data(symbol, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        return data
    except ValueError as e:
        print(f"Greška u dohvatanju podataka: {e}")
        return None

def prepare_features(data):
    # Obrada, priprema i izračunavanje podataka pokretnog proseka
    data['moving_average'] = data['4. close'].rolling(window=5).mean()
    # Model kropisti 'moving_average' i '4. close' kao ulaz
    return data[['4. close', 'moving_average']].iloc[-1].values.reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    api_key = 'FR3Z1V1HK57KDTCQ'
    stock_data = get_stock_data(symbol, api_key)
    if stock_data is not None:
        features = prepare_features(stock_data)
        prediction = model.predict(features)
        return render_template('index.html', prediction=prediction[0])
    else:
        error_message = "Nije moguće dobiti podatke za simbol: " + symbol
        return render_template('index.html', error_message=error_message)
    
if __name__ == '__main__':
    app.run(debug=True)
