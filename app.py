import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

@st.cache_data
def load_data():
    data = pd.read_csv('house_data.csv')
    return data

def train_model(data):
    if 'statezip' in data.columns:
        data = data.drop(columns=['statezip'])
    
    if 'city' in data.columns:
        data = data.drop(columns=['city'])
    
    if 'date' in data.columns:
        data = data.drop(columns=['date'])

    if 'street' in data.columns:
        data = data.drop(columns=['street'])

    if 'country' in data.columns:
        data = data.drop(columns=['country'])

    data = pd.get_dummies(data, drop_first=True)

    data = data.fillna(data.mean())

    X = data.drop(columns=['price'])
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, scaler, X_train.columns, mae, rmse, r2, y_test, y_pred

def predict_price(model, scaler, input_data, columns):
    input_df = pd.DataFrame([input_data], columns=columns)
    
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    
    missing_cols = set(columns) - set(input_df_encoded.columns)
    for col in missing_cols:
        input_df_encoded[col] = 0

    input_df_encoded = input_df_encoded[columns]
    
    imputer = SimpleImputer(strategy='mean')
    input_df_encoded = pd.DataFrame(imputer.fit_transform(input_df_encoded), columns=input_df_encoded.columns)

    input_scaled = scaler.transform(input_df_encoded)

    predicted_price = model.predict(input_scaled)

    return predicted_price[0]

def plot_feature_importances(model, X):
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_features = X.columns[sorted_idx]
    sorted_importances = feature_importances[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    st.pyplot(plt)

def plot_predictions_vs_actual(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    st.pyplot(plt)

def plot_error_distribution(y_test, y_pred):
    errors = y_pred - y_test
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='blue', alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    st.pyplot(plt)

def main():
    st.title('House Price Prediction')

    data = load_data()
    model, scaler, columns, mae, rmse, r2, y_test, y_pred = train_model(data)

    st.subheader('Model Evaluation')
    st.write(f'Mean Absolute Error (MAE): {mae}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse}')
    st.write(f'R-squared (R²): {r2}')

    st.subheader('Model Visualizations')
    plot_feature_importances(model, data.drop(columns=['price']))
    plot_predictions_vs_actual(y_test, y_pred)
    plot_error_distribution(y_test, y_pred)

    st.subheader('Enter the details of the house to predict the price')
    
    bedrooms = st.number_input('Bedrooms', min_value=1, max_value=10, value=3)
    bathrooms = st.number_input('Bathrooms', min_value=1, max_value=5, value=2)
    sqft_living = st.number_input('Sqft Living', min_value=500, max_value=10000, value=2000)
    sqft_lot = st.number_input('Sqft Lot', min_value=500, max_value=50000, value=5000)
    floors = st.number_input('Floors', min_value=1, max_value=3, value=1)
    waterfront = st.selectbox('Waterfront', options=[0, 1], index=0)
    view = st.selectbox('View', options=[0, 1, 2, 3, 4], index=0)
    condition = st.selectbox('Condition', options=[1, 2, 3, 4, 5], index=2)
    sqft_above = st.number_input('Sqft Above', min_value=500, max_value=5000, value=1500)
    sqft_basement = st.number_input('Sqft Basement', min_value=0, max_value=5000, value=0)
    yr_built = st.number_input('Year Built', min_value=1900, max_value=2025, value=1990)
    yr_renovated = st.number_input('Year Renovated', min_value=0, max_value=2025, value=0)

    inputs = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated
    }

    if st.button('Predict Price'):
        predicted_price = predict_price(model, scaler, inputs, columns)
        st.write(f'Predicted Price: ${predicted_price:,.2f}')

if __name__ == '__main__':
    main()
