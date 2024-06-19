import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# Function to load data and process it
def load_data(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        st.error(f"Could not find the file: {filename}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Function to plot scatter plot and return the figure
def plot_scatter(x, y, title):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(title)
    return fig

# Function to display data in a table below the plot
def display_data_table(data, title):
    st.write(f"### {title}")
    st.dataframe(data)

# Main Streamlit app
def main():
    st.title("Car Price Prediction")

    # Sidebar for uploading files
    st.sidebar.title("Upload Files")
    training_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type=['csv'])
    testing_file = st.sidebar.file_uploader("Upload Testing Data (CSV)", type=['csv'])

    if training_file and testing_file:
        st.sidebar.info("Files uploaded successfully")

        # Load training and testing data
        train_data = load_data(training_file)
        test_data = load_data(testing_file)

        # Display data and basic info
        st.write("### Training Dataframe:")
        st.dataframe(train_data.head())
        st.write("### Training Data Info:")
        st.write(train_data.info())

        st.write("### Testing Dataframe:")
        st.dataframe(test_data.head())
        st.write("### Testing Data Info:")
        st.write(test_data.info())

        # Encoding categorical columns
        train_data = pd.get_dummies(train_data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)
        test_data = pd.get_dummies(test_data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

        # Splitting the data and Target
        X_train = train_data.drop(['Car_Name', 'Selling_Price'], axis=1)
        Y_train = train_data['Selling_Price']

        X_test = test_data.drop(['Car_Name', 'Selling_Price'], axis=1)
        Y_test = test_data['Selling_Price']

        # Model Training and Evaluation - Linear Regression
        lin_reg_model = LinearRegression()
        lin_reg_model.fit(X_train, Y_train)

        training_data_prediction = lin_reg_model.predict(X_train)
        train_error = metrics.r2_score(Y_train, training_data_prediction)

        st.write("### Training Results with Linear Regression:")
        fig_train = plot_scatter(Y_train, training_data_prediction, "Actual Price vs Predicted Price (Training)")
        st.pyplot(fig_train)
        display_data_table(pd.DataFrame({'Actual Price': Y_train, 'Predicted Price': training_data_prediction}), "Training Results with Linear Regression")

        test_data_prediction = lin_reg_model.predict(X_test)
        test_error = metrics.r2_score(Y_test, test_data_prediction)

        st.write("### Testing Results:")
        fig_test = plot_scatter(Y_test, test_data_prediction, "Actual Price vs Predicted Price (Testing)")
        st.pyplot(fig_test)
        display_data_table(pd.DataFrame({'Actual Price': Y_test, 'Predicted Price': test_data_prediction}), "Testing Results")

        # Model Training and Evaluation - Lasso Regression
        lasso_reg_model = Lasso()
        lasso_reg_model.fit(X_train, Y_train)

        training_data_prediction_lasso = lasso_reg_model.predict(X_train)
        train_error_lasso = metrics.r2_score(Y_train, training_data_prediction_lasso)

        st.write("### Training Results with Lasso Regression:")
        fig_train_lasso = plot_scatter(Y_train, training_data_prediction_lasso, "Actual Price vs Predicted Price (Training)")
        st.pyplot(fig_train_lasso)
        display_data_table(pd.DataFrame({'Actual Price': Y_train, 'Predicted Price': training_data_prediction_lasso}), "Training Results with Lasso Regression")

        test_data_prediction_lasso = lasso_reg_model.predict(X_test)
        test_error_lasso = metrics.r2_score(Y_test, test_data_prediction_lasso)

        st.write("### Testing Results:")
        fig_test_lasso = plot_scatter(Y_test, test_data_prediction_lasso, "Actual Price vs Predicted Price (Testing)")
        st.pyplot(fig_test_lasso)
        display_data_table(pd.DataFrame({'Actual Price': Y_test, 'Predicted Price': test_data_prediction_lasso}), "Testing Results")

        # Closing figures to release resources (optional)
        plt.close(fig_train)
        plt.close(fig_test)
        plt.close(fig_train_lasso)
        plt.close(fig_test_lasso)

    else:
        st.info("Please upload CSV files.")

if __name__ == '__main__':
    main()
