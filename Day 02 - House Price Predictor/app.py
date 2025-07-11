import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


data = pd.read_csv('housing.csv')


data['total_bedrooms'] = data['total_bedrooms'].fillna(np.mean(data['total_bedrooms']))
data.drop('ocean_proximity', axis=1, inplace=True)


X = data.iloc[:, data.columns != 'median_house_value']
Y = data['median_house_value']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


st.title("House Price Prediction")


st.header("Enter the Features")



col1, col2, col3 = st.columns(3)



with col1:
    longitude = st.number_input('Longitude', min_value=float(X['longitude'].min()), max_value=float(X['longitude'].max()), value=float(X['longitude'].mean()))
    housing_median_age = st.number_input('Housing Median Age', min_value=int(X['housing_median_age'].min()), max_value=int(X['housing_median_age'].max()), value=int(X['housing_median_age'].mean()))
    total_rooms = st.number_input('Total Rooms', min_value=int(X['total_rooms'].min()), max_value=int(X['total_rooms'].max()), value=int(X['total_rooms'].mean()))
    population = st.number_input('Population', min_value=int(X['population'].min()), max_value=int(X['population'].max()), value=int(X['population'].mean()))


with col2:
    latitude = st.number_input('Latitude', min_value=float(X['latitude'].min()), max_value=float(X['latitude'].max()), value=float(X['latitude'].mean()))
    total_bedrooms = st.number_input('Total Bedrooms', min_value=int(X['total_bedrooms'].min()), max_value=int(X['total_bedrooms'].max()), value=int(X['total_bedrooms'].mean()))
    households = st.number_input('Households', min_value=int(X['households'].min()), max_value=int(X['households'].max()), value=int(X['households'].mean()))
    median_income = st.number_input('Median Income', min_value=float(X['median_income'].min()), max_value=float(X['median_income'].max()), value=float(X['median_income'].mean()))



input_data = {
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': total_rooms,
    'total_bedrooms': total_bedrooms,
    'population': population,
    'households': households,
    'median_income': median_income
}



user_input = pd.DataFrame(input_data, index=[0])



regression = LinearRegression()
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)


pol = PolynomialFeatures(degree=2)
X_train_p = pol.fit_transform(X_train)
X_test_p = pol.transform(X_test)

regression_p = LinearRegression()
regression_p.fit(X_train_p, y_train)
y_pred_p = regression_p.predict(X_test_p)


# Predict using both models on user input
user_input_p = pol.transform(user_input)
prediction_lr = regression.predict(user_input)
prediction_pr = regression_p.predict(user_input_p)



with col3:
    st.subheader("Predicted House Price:")
    st.write(f"Linear Regression Prediction: ${prediction_lr[0]:,.2f}")
    st.write(f"Polynomial Regression Prediction: ${prediction_pr[0]:,.2f}")

  
    st.subheader("Model Performance")
    st.write("R2 Score for Linear Regression:", r2_score(y_test, y_pred))
    st.write("R2 Score for Polynomial Regression:", r2_score(y_test, y_pred_p))



fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_test.values, label="True Values", color='blue')
ax.plot(y_pred, label="Linear Regression", color='green')
ax.plot(y_pred_p, label="Polynomial Regression", color='red')
ax.legend(loc='upper right')
st.pyplot(fig)
