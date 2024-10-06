
# Bike Share Dashboard

Dashboard ini menampilkan visualisasi data peminjaman sepeda berdasarkan dataset `hour.csv`. Anda dapat melihat berbagai analisis dan hubungan antara faktor cuaca, musim, dan waktu dengan jumlah peminjaman sepeda.

## Fitur Dashboard

- Menampilkan dataset dan ringkasan statistik
- Visualisasi rata-rata peminjaman sepeda pada hari kerja dan hari libur (casual dan registered)
- Korelasi antara situasi cuaca, musim, jam, kecepatan angin, suhu, dan kelembapan dengan jumlah peminjaman sepeda

## Requirements

Sebelum menjalankan aplikasi ini, pastikan Anda telah menginstal beberapa dependensi yang dibutuhkan:

1. Python 3.x
2. Library Python:
   - `streamlit`
   - `pandas`
   - `numpy`
   - `plotly`
   - `seaborn`
   - `matplotlib`

### Instalasi Dependensi

Anda dapat menginstal dependensi yang diperlukan dengan menjalankan perintah berikut:

```bash
pip install streamlit pandas numpy plotly seaborn matplotlib
```

## Menjalankan Dashboard

Ikuti langkah-langkah berikut untuk menjalankan aplikasi dashboard ini:

1. **Clone repository** atau copy file dashboard ke dalam folder lokal Anda.
2. **Simpan file dataset `hour.csv`** di direktori yang sama dengan file Python.
3. Buka terminal/command prompt dan arahkan ke direktori tempat file tersebut berada.
4. Jalankan perintah berikut untuk menjalankan dashboard:

   ```bash
   streamlit run nama_file.py
   ```

   Gantilah `nama_file.py` dengan nama file Python Anda (misalnya, `bike_share_dashboard.py`).

5. Dashboard akan terbuka di browser default Anda pada alamat:

   ```bash
   http://localhost:8501
   ```

## Penggunaan

- **Tampilkan Dataset:** Centang opsi "Tampilkan Dataset" di sidebar untuk melihat dataset asli.
- **Tampilkan Ringkasan Dataset:** Centang opsi "Tampilkan Ringkasan Dataset" di sidebar untuk melihat deskripsi statistik dataset.
- **Rata-rata Peminjaman Sepeda:** Lihat rata-rata peminjaman sepeda pada hari kerja dan hari libur.
- **Korelasi dan Visualisasi:** Lihat visualisasi hubungan antara cuaca, musim, jam, dan faktor lainnya dengan jumlah peminjaman sepeda.

## Penulis

- **Nama:** Andes Potipera Sitepu
- **Email:** andessitepu221204@gmail.com
- **Dicoding:** andes_sitepu
