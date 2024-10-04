# Analisis Data dengan Python : Bike Sharing Dataset
Proyek Akhir dari "Belajar Analisis Data dengan Python" pada Platform Dicoding.
Proyek analisis data ini membahas dataset tentang peminjaman sepeda dan pengaruh dari variabel seperti suhu, kelembapan, kencang angin, dan situasi cuaca terhadap jumlah peminjaman sepeda.
Dataset ini terdiri atas beberapa jenis data yaitu:

Data Numerik:

- Temp (Suhu): Merupakan suhu yang dinormalisasi dalam skala Celsius. Nilainya dibagi menjadi 41 (maksimal).
- Atemp (Suhu Perasaan): Menyatakan suhu perasaan yang dinormalisasi dalam skala Celsius. Nilainya dibagi menjadi 50 (maksimal).
- Hum (Kelembaban): Merupakan tingkat kelembaban yang dinormalisasi. Nilainya dibagi menjadi 100 (maksimal).
- Windspeed (Kecepatan Angin): Menunjukkan kecepatan angin yang dinormalisasi. Nilainya dibagi menjadi 67 (maksimal).
- Casual (Pengguna Sepeda Sewaan Kasual): Menyatakan jumlah pengguna sepeda sewaan kasual.
- Registered (Pengguna Sepeda Sewaan Terdaftar): Menunjukkan jumlah pengguna sepeda sewaan yang terdaftar.
- Cnt (Total Peminjaman Sepeda): Merupakan jumlah total sepeda yang disewakan, termasuk pengguna kasual dan terdaftar.

Data Kategorikal:

- Season: Merupakan informasi tentang musim dalam tahun (1: musim semi, 2: musim panas, 3: musim gugur, 4: musim dingin).
- Year (Tahun): Menunjukkan tahun (0: 2011, 1: 2012).
- Month (Bulan): Memberikan informasi tentang bulan dalam setahun (1 hingga 12).
- Hour (Jam): Menunjukkan jam dalam sehari (0 hingga 23).
- Holiday (Hari Libur): Mengindikasikan apakah hari itu merupakan hari libur (1 jika ya, 0 jika tidak).
- Weekday (Hari dalam Seminggu): Menyatakan hari dalam seminggu.
- Workingday (Hari Kerja): Menunjukkan apakah hari itu merupakan hari kerja (1 jika ya, 0 jika tidak).
- Weathersit (Kondisi Cuaca): Memberikan informasi tentang kondisi cuaca (1: Cerah, sedikit awan, sebagian cerah, 2: Berkabut + mendung, berkabut + awan rusak, berkabut + sebagian cerah, berkabut, 3: Hujan ringan, petir, awan terpencar, hujan ringan + awan terpencar, 4: Hujan lebat, hujan es + petir + kabut, salju + kabut).

## Langkah Pengerjaan Proyek
- Menentukan Pertanyaan Bisnis
- Import Library
- Data Wrangling
- Exploratory Data Analysis
- Visualization & Explanatory Analysis
- Conclusion
- Membuat Dashboard sederhana dengan Streamlit

## Documentation
![image](https://github.com/Natuman-P/KECERDASAN-BUATAN-2024-ANDES/blob/main/output.png)

![image](https://github.com/Natuman-P/KECERDASAN-BUATAN-2024-ANDES/blob/main/output2.png)


## Kesimpulan
**Conclution pertanyaan 1 : Bagaimana jam, suhu, musim, kelembapan memengaruhi pola penggunaan sepeda setiap harinya?**

Berdasarkan analisis yang telah dilakukan terhadap pengaruh varibel-variabel tersebut dengan jumlah peminjaman sepeda dapat disimpulkan bahwa orang-orang lebih menyukai kondisi cuaca yang lebih baik untuk menggunakan sepeda seperti cuaca yang cerah dengan kelembapan yang rendah. Orang-orang juga menggunakan sepeda untuk pergi berangkat ke tempat kerja mereka karena terjadi kenaikan jumlah peminjaman sepeda menjelang jam masuk kerja dan jam pulang kerja.

**Conclution pertanyaan 2 : Bagaimana perbedaan hari peminjaman memengaruhi pola penggunaan sepeda?**

Berdasarkan perhitungan yang telah dilakukan untuk melihat perbedaan dari jumlah peminjaman pada kedua hari tersebut dapat disimpulkan bahwa jumlah peminjaman sepeda lebih banyak terjadi pada hari kerja, hal ini terjadi karena orang-orang lebih sering menggunakan sepeda untuk berangkat atau pulang dari tempat kerja mereka yang dimana terjadi pada hari kerja, hal ini didasari pada analisis untuk pertanyaan 1 yaitu pada bagian "pengaruh jam terhadap jumlah peminjaman sepeda" dimana kenaikan jumlah peminjaman sepeda terjadi menjelenag jam masuk kerja dan jam pulang kerja.
