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
    ax.set_xlabel("Harga yang sebenarnya")
    ax.set_ylabel("Harga yang di prediksi")
    ax.set_title(title)
    return fig

# Function to display data in a table below the plot
def display_data_table(data, title):
    st.write(f"### {title}")
    st.write(data)

# Main Streamlit app
def main():
    st.title("Prediksi Harga Mobil")

    # Sidebar for uploading files
    st.sidebar.title("Upload Files")
    training_file = st.sidebar.file_uploader("Upload Data Latihan (CSV)", type=['csv'])
    testing_file = st.sidebar.file_uploader("Upload Data Test (CSV)", type=['csv'])

    if training_file and testing_file:
        st.sidebar.info("File berhasil di upload")

        # Load training and testing data
        train_data = load_data(training_file)
        test_data = load_data(testing_file)

        # Display data and basic info
        st.write("### Dataframe data latihan:")
        st.write(df)
        st.write("### Data Latihan (5 data pertama) :")
        st.write(train_data.head())
        st.write("### Bentuk Data Frame :")
        st.write(train_data.shape)
        st.write("### Data Frame Info :")
        st.write(train_data.info())
        st.write("### Nilai yang tidak ada atau hilang :")
        st.write(train_data.isnull().sum())

        st.write("### Dataframe data uji coba / Testing:")
        st.write(df)
        st.write("### Data uji coba / Testing (5 data pertama) :")
        st.write(test_data.head())
        st.write("### Bentuk Data Frame :")
        st.write(test_data.shape)
        st.write("### Data Frame Info :")
        st.write(test_data.info())
        st.write("### Nilai yang tidak ada atau hilang :")
        st.write(test_data.isnull().sum())

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

        st.write("### Grafik Hasil Latihan dengan Linear Regression:")
        fig_train = plot_scatter(Y_train, training_data_prediction, "Harga yang sebenarnya vs harga Harga yang di prediksi (Latihan)")
        st.pyplot(fig_train)
        display_data_table(pd.DataFrame({'Harga yang sebenarnya': Y_train, 'Harga yang di prediksi': training_data_prediction}), "Hasil Latihan dengan Linear Regression")

        test_data_prediction = lin_reg_model.predict(X_test)
        test_error = metrics.r2_score(Y_test, test_data_prediction)

        st.write("### Grafik Hasil Testing:")
        fig_test = plot_scatter(Y_test, test_data_prediction, "Harga yang sebenarnya vs harga Harga yang di prediksi (Testing)")
        st.pyplot(fig_test)
        display_data_table(pd.DataFrame({'Harga yang sebenarnya': Y_test, 'Harga yang di prediksi': test_data_prediction}), "Hasil Testing")

        # Model Training and Evaluation - Lasso Regression
        lasso_reg_model = Lasso()
        lasso_reg_model.fit(X_train, Y_train)

        training_data_prediction_lasso = lasso_reg_model.predict(X_train)
        train_error_lasso = metrics.r2_score(Y_train, training_data_prediction_lasso)

        st.write("### Grafik Hasil Latihan dengan Lasso Regression:")
        fig_train_lasso = plot_scatter(Y_train, training_data_prediction_lasso, "Harga yang sebenarnya vs harga Harga yang di prediksi (Latihan)")
        st.pyplot(fig_train_lasso)
        display_data_table(pd.DataFrame({'Harga yang sebenarnya': Y_train, 'Harga yang di prediksi': training_data_prediction_lasso}), "Hasil Latihan dengan Lasso Regression")

        test_data_prediction_lasso = lasso_reg_model.predict(X_test)
        test_error_lasso = metrics.r2_score(Y_test, test_data_prediction_lasso)

        st.write("### Grafik Hasil Testing:")
        fig_test_lasso = plot_scatter(Y_test, test_data_prediction_lasso, "Harga yang sebenarnya vs harga Harga yang di prediksi (Testing)")
        st.pyplot(fig_test_lasso)
        display_data_table(pd.DataFrame({'Harga yang sebenarnya': Y_test, 'Harga yang di prediksi': test_data_prediction_lasso}), "Hasil Testing")

        # Closing figures to release resources (optional)
        plt.close(fig_train)
        plt.close(fig_test)
        plt.close(fig_train_lasso)
        plt.close(fig_test_lasso)

    else:
        st.info("Tolong upload data CSV.")

if __name__ == '__main__':
    main()
