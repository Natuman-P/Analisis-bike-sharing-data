import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics
from PIL import Image

# Load data
@st.cache
def load_data():
    car_dataset = pd.read_csv('data.csv')
    return car_dataset

# Encode categorical data
def encode_categorical_data(car_dataset):
    car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}}, inplace=True)
    car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}}, inplace=True)
    car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}}, inplace=True)
    return car_dataset

# Model training and evaluation
def train_and_evaluate_model(X_train, X_test, Y_train, Y_test):
    # Linear Regression
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, Y_train)
    
    # Prediction on training data
    training_data_prediction = lin_reg_model.predict(X_train)
    r2_train = metrics.r2_score(Y_train, training_data_prediction)
    
    # Prediction on test data
    test_data_prediction = lin_reg_model.predict(X_test)
    r2_test = metrics.r2_score(Y_test, test_data_prediction)
    
    return lin_reg_model, r2_train, r2_test

# Visualization function
def visualize_results(Y_train, training_data_prediction, Y_test, test_data_prediction, model_name):
    # Plotting actual vs predicted
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(Y_train, training_data_prediction)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{model_name} - Training Set")

    plt.subplot(1, 2, 2)
    plt.scatter(Y_test, test_data_prediction)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{model_name} - Test Set")

    plt.tight_layout()

    # Save plot to a temporary file
    plt.savefig("plot.png")
    image = Image.open("plot.png")
    st.image(image, caption=f"Actual vs Predicted Prices - {model_name}", use_column_width=True)

# Streamlit App
def main():
    st.title("Car Price Prediction")
    st.sidebar.title("Options")
    
    car_dataset = load_data()
    car_dataset = encode_categorical_data(car_dataset)
    
    X = car_dataset.drop(['Car_Name','Selling_Price'], axis=1)
    Y = car_dataset['Selling_Price']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
    
    model_name = st.sidebar.selectbox("Select Model", ["Linear Regression", "Lasso Regression"])
    
    if model_name == "Linear Regression":
        model, r2_train, r2_test = train_and_evaluate_model(X_train, X_test, Y_train, Y_test)
    elif model_name == "Lasso Regression":
        model = Lasso()
        model.fit(X_train, Y_train)
        training_data_prediction = model.predict(X_train)
        test_data_prediction = model.predict(X_test)
        r2_train = metrics.r2_score(Y_train, training_data_prediction)
        r2_test = metrics.r2_score(Y_test, test_data_prediction)
    
    st.write(f"## {model_name} Model Evaluation")
    st.write(f"R-squared (Training Set): {r2_train}")
    st.write(f"R-squared (Test Set): {r2_test}")
    
    visualize_results(Y_train, training_data_prediction, Y_test, test_data_prediction, model_name)

if __name__ == "__main__":
    main()
