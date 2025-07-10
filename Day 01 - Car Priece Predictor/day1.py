import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# Load the dataset (ensure the path is correct or adjust it accordingly)
dataset = pd.read_csv('processes2.csv')

# Define X and Y
X = dataset.iloc[:, dataset.columns != 'selling_price']  # Exclude the target column
Y = dataset['selling_price']

# OneHotEncoding for categorical features
ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int64, handle_unknown='ignore')

# Fill missing values with 'Unknown' in categorical columns to prevent errors
X[['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']] = X[['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']].fillna('Unknown')

# Convert categorical columns to string type
X[['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']] = X[['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']].astype(str)

# Apply OneHotEncoder to transform categorical features
X_train_new = ohe.fit_transform(X[['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']])

# Preprocessing
X_train = np.hstack((X[['year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)']].values, X_train_new))

# Standardizing the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, Y)

# Streamlit interface
st.title('Machine Learning Project Day 01')
st.title('Car Price Prediction')
st.markdown("Enter the car details below to get the predicted selling price.")

# Create two columns for layout
col1, col2 = st.columns(2)

# Inputs from the user - placed in two columns
with col1:
    name = st.text_input("Car Name", "")
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    owner = st.number_input("Number of Owners", min_value=1, max_value=5)
    year = st.number_input("Year of Manufacture", min_value=2000, max_value=2023)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000)

with col2:
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
    mileage_unit = st.selectbox("Mileage Unit", ["km/ltr", "km/kg"])
    seats = st.number_input("Number of Seats", min_value=2, max_value=8)
    max_power = st.number_input("Max Power (in BHP)", min_value=20, max_value=1000)
    mileage = st.number_input("Mileage", min_value=1.0, max_value=50.0)
    engine_cc = st.number_input("Engine Capacity (in CC)", min_value=600, max_value=6000)

# Make prediction when user clicks 'Predict'
if st.button("Predict Price"):
    # Create a DataFrame from the user input
    user_input = pd.DataFrame([[name, fuel, seller_type, transmission, owner, mileage_unit, year, km_driven, seats, max_power, mileage, engine_cc]],
                              columns=['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit', 'year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)'])

    # Fill missing values with 'Unknown' in user input data to prevent errors
    user_input[['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']] = user_input[['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']].fillna('Unknown')

    # Convert categorical data to string
    user_input[['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']] = user_input[['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']].astype(str)

    # Process the categorical data using OneHotEncoder
    user_input_new = ohe.transform(user_input[['name', 'fuel', 'seller_type', 'transmission', 'owner', 'Mileage Unit']])

    # Process numerical data and combine with encoded categorical data
    user_input_data = np.hstack((user_input[['year', 'km_driven', 'seats', 'max_power (in bph)', 'Mileage', 'Engine (CC)']].values, user_input_new))

    # Standardize the input
    user_input_scaled = sc.transform(user_input_data)

    # Predict the selling price
    predicted_price = regressor.predict(user_input_scaled)

    # Display the prediction
  st.write(f'The predicted selling price of the car is: RS{predicted_price[0]:,.2f}')
