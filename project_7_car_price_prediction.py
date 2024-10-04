import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_resource
def read_data():
    hour_df = pd.read_csv('data/hour.csv')

    categorical_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    hour_df[categorical_columns] = hour_df[categorical_columns].astype('category')

    def impute_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        data[column] = data[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

    columns_hour = ['hum', 'windspeed', 'casual', 'registered', 'cnt']
    for column in columns_hour:
        impute_outliers(hour_df, column)

    return hour_df


hour_df = read_data()

st.title("Bike Share Dashboard")

st.sidebar.title("Author:")
st.sidebar.markdown("**• Nama: Andes Potipera Sitepu**")
st.sidebar.markdown("**• Email: andessitepu221204@gmail.com**")
st.sidebar.markdown("**• Dicoding: andes_sitepu**")

if st.sidebar.checkbox("Tampilkan Dataset"):
    st.subheader("Dataset Original")
    st.write(hour_df)

if st.sidebar.checkbox("Tampilkan Ringkasan Dataset"):
    st.subheader("Deskripsi Statistik Dataset")
    st.write(hour_df.describe())

st.sidebar.markdown('**Season:**')
st.sidebar.markdown('1: Spring')
st.sidebar.markdown('2: Summer')
st.sidebar.markdown('1: Fall')
st.sidebar.markdown('1: Winter')

st.sidebar.markdown('**Weather Situation:**')
st.sidebar.markdown('1: Clear, Few clouds, Partly cloudy, Partly cloudy')
st.sidebar.markdown('2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist')
st.sidebar.markdown('3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds')
st.sidebar.markdown('4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog')

mean_rental_casual_weekend = hour_df[hour_df['holiday'] == 1]['casual'].mean()
mean_rental_registered_weekend = hour_df[hour_df['holiday'] == 1]['registered'].mean()


mean_rental_casual_workingday = hour_df[hour_df['workingday'] == 1]['casual'].mean()  
mean_rental_registered_workingday = hour_df[hour_df['workingday'] == 1]['registered'].mean()  

text_weekday = f"Rata-rata peminjaman sepeda casual pada hari kerja: {mean_rental_casual_workingday:.2f}\nRata-rata peminjaman sepeda registered pada hari kerja: {mean_rental_registered_workingday:.2f}"
text_weekend = f"Rata-rata peminjaman sepeda casual pada hari libur: {mean_rental_casual_weekend:.2f}\nRata-rata peminjaman sepeda registered pada hari libur: {mean_rental_registered_weekend:.2f}"

fig1 = go.Figure(data=[
    go.Bar(x=['Casual', 'Registered'], y=[mean_rental_casual_workingday, mean_rental_registered_workingday], marker_color=['blue', 'orange'])
])
fig1.update_layout(
    title='Average Bike Rentals on Weekdays',
    xaxis_title='Rental Type',
    yaxis_title='Average Rentals',
    annotations=[dict(x=xi, y=yi, text=f'{yi:.2f}', showarrow=False) for xi, yi in zip(['Casual', 'Registered'], [mean_rental_casual_workingday, mean_rental_registered_workingday])]
)

fig2 = go.Figure(data=[
    go.Bar(x=['Casual', 'Registered'], y=[mean_rental_casual_weekend, mean_rental_registered_weekend], marker_color=['green', 'red'])
])
fig2.update_layout(
    title='Average Bike Rentals on Weekends',
    xaxis_title='Rental Type',
    yaxis_title='Average Rentals',
    annotations=[dict(x=xi, y=yi, text=f'{yi:.2f}', showarrow=False) for xi, yi in zip(['Casual', 'Registered'], [mean_rental_casual_weekend, mean_rental_registered_weekend])]
)

st.plotly_chart(fig1, use_container_width=True)
st.write(text_weekday)
st.plotly_chart(fig2, use_container_width=True)
st.write(text_weekend)

col1, col2 = st.columns(2)

with col1:

    st.subheader("Korelasi Situasi Cuaca dan Rental Sepeda")

    weather_count_df = hour_df.groupby("weathersit")["cnt"].sum().reset_index()
    fig_weather_count = px.bar(weather_count_df, x="weathersit",
                               y="cnt", title="Situasi Cuaca terhadap Jumlah Rental Sepeda")
    fig_weather_count.update_traces(marker_color='skyblue')

    st.plotly_chart(fig_weather_count, use_container_width=True, height=400, width=800)

with col2:
    
    st.subheader("Korelasi Musim dan Rental Sepeda")

    season_count_df = hour_df.groupby("season")["cnt"].sum().reset_index()
    fig_season_count = px.bar(season_count_df, x="season",
                              y="cnt", title="Musim terhadap Jumlah Rental Sepeda")
    fig_season_count.update_traces(marker_color='salmon')
    
    st.plotly_chart(fig_season_count, use_container_width=True, height=400, width=600)

plt.figure(figsize=(14, 24))

st.subheader("Korelasi Jam, Kecepatan Angin, Suhu dan Kelembapan terhadap Jumlah Rental Sepeda")

plt.subplot(6, 1, 1)
sns.lineplot(data=hour_df, x='hr', y='cnt', color='magenta')
plt.title('Pengaruh Hour Terhadap Jumlah Peminjaman Sepeda')
plt.xlabel('Hour')
plt.ylabel('Jumlah Peminjaman Sepeda')
plt.xticks(np.arange(1, 25, 1)) 

fig = plt.gcf()


st.pyplot(fig)


plt.figure(figsize=(14, 24))
plt.subplot(6, 1, 2)
sns.lineplot(data=hour_df, x='windspeed', y='cnt', color='red')
plt.title('Pengaruh Windspeed Terhadap Jumlah Peminjaman Sepeda')
plt.xlabel('Windspeed')
plt.ylabel('Jumlah Peminjaman Sepeda')
fig = plt.gcf()
st.pyplot(fig)


plt.figure(figsize=(14, 24))
plt.subplot(6, 1, 3)
sns.lineplot(data=hour_df, x='temp', y='cnt', color='blue')
plt.title('Pengaruh Temperature Terhadap Jumlah Peminjaman Sepeda')
plt.xlabel('Temperature')
plt.ylabel('Jumlah Peminjaman Sepeda')
fig = plt.gcf()
st.pyplot(fig)


plt.figure(figsize=(14, 24))
plt.subplot(6, 1, 4)
sns.lineplot(data=hour_df, x='hum', y='cnt', color='orange')
plt.title('Pengaruh Humidity Terhadap Jumlah Peminjaman Sepeda')
plt.xlabel('Humidity')
plt.ylabel('Jumlah Peminjaman Sepeda')
fig = plt.gcf()
st.pyplot(fig)
