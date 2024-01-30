from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def get_stock_data(symbol, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')
    return data

def prepare_data(data):
    data['moving_average'] = data['4. close'].rolling(window=5).mean()
    data['target'] = data['4. close'].shift(-1)  # Ciljna promenljiva (sutrašnja cena)
    data = data.dropna()  # Uklanjanje redova sa NaN vrednostima
    return data[['4. close', 'moving_average']], data['target']


api_key = 'FR3Z1V1HK57KDTCQ'
symbol = 'AAPL'  # Primer simbola akcije

# Preuzimanje podataka
stock_data = get_stock_data(symbol, api_key)

# Priprema podataka
X, y = prepare_data(stock_data)

# Podela podataka na trening i test skupove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kreiranje i treniranje modela
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Čuvanje modela kao pkl fajl
joblib.dump(model, 'model_random_forest.pkl')
