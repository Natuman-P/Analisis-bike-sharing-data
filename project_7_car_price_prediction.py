import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# loading the data from csv file to pandas dataframe
car_dataset = pd.read_csv('data.csv')

# inspecting the first 5 rows of the dataframe
st.write(car_dataset.head())

# checking the number of rows and columns
st.write("Shape of dataset:", car_dataset.shape)

# getting some information about the dataset
st.write("Information about dataset:")
st.write(car_dataset.info())

# checking the number of missing values
st.write("Number of missing values:")
st.write(car_dataset.isnull().sum())

# checking the distribution of categorical data
st.write("Distribution of Fuel_Type:")
st.write(car_dataset['Fuel_Type'].value_counts())
st.write("Distribution of Seller_Type:")
st.write(car_dataset['Seller_Type'].value_counts())
st.write("Distribution of Transmission:")
st.write(car_dataset['Transmission'].value_counts())

# Encoding categorical columns
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
                     'Seller_Type': {'Dealer': 0, 'Individual': 1},
                     'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

st.write("Encoded dataset:")
st.write(car_dataset.head())

# Splitting the data and Target
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

st.write("Features (X):")
st.write(X.head())
st.write("Target (Y):")
st.write(Y.head())

# Splitting Training and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# Model Training - Linear Regression
st.write("### Linear Regression Model")

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# Model Evaluation - Linear Regression
training_data_prediction = lin_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
st.write("R squared Error (Train) - Linear Regression: ", error_score)

st.write("Actual vs Predicted Prices (Train) - Linear Regression:")
train_results = pd.DataFrame({'Actual': Y_train, 'Predicted': training_data_prediction})
st.write(train_results)

# Scatter plot for Training data - Linear Regression
st.write("Scatter plot of Actual vs Predicted Prices (Train) - Linear Regression:")
st.write(train_results.plot(kind='scatter', x='Actual', y='Predicted'))

test_data_prediction = lin_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
st.write("R squared Error (Test) - Linear Regression: ", error_score)

st.write("Actual vs Predicted Prices (Test) - Linear Regression:")
test_results = pd.DataFrame({'Actual': Y_test, 'Predicted': test_data_prediction})
st.write(test_results)

# Scatter plot for Test data - Linear Regression
st.write("Scatter plot of Actual vs Predicted Prices (Test) - Linear Regression:")
st.write(test_results.plot(kind='scatter', x='Actual', y='Predicted'))

# Model Training - Lasso Regression
st.write("### Lasso Regression Model")

lasso_reg_model = Lasso()
lasso_reg_model.fit(X_train, Y_train)

# Model Evaluation - Lasso Regression
training_data_prediction_lasso = lasso_reg_model.predict(X_train)
error_score_lasso = metrics.r2_score(Y_train, training_data_prediction_lasso)
st.write("R squared Error (Train) - Lasso Regression: ", error_score_lasso)

st.write("Actual vs Predicted Prices (Train) - Lasso Regression:")
train_results_lasso = pd.DataFrame({'Actual': Y_train, 'Predicted': training_data_prediction_lasso})
st.write(train_results_lasso)

# Scatter plot for Training data - Lasso Regression
st.write("Scatter plot of Actual vs Predicted Prices (Train) - Lasso Regression:")
st.write(train_results_lasso.plot(kind='scatter', x='Actual', y='Predicted'))

test_data_prediction_lasso = lasso_reg_model.predict(X_test)
error_score_lasso = metrics.r2_score(Y_test, test_data_prediction_lasso)
st.write("R squared Error (Test) - Lasso Regression: ", error_score_lasso)

st.write("Actual vs Predicted Prices (Test) - Lasso Regression:")
test_results_lasso = pd.DataFrame({'Actual': Y_test, 'Predicted': test_data_prediction_lasso})
st.write(test_results_lasso)

# Scatter plot for Test data - Lasso Regression
st.write("Scatter plot of Actual vs Predicted Prices (Test) - Lasso Regression:")
st.write(test_results_lasso.plot(kind='scatter', x='Actual', y='Predicted'))
