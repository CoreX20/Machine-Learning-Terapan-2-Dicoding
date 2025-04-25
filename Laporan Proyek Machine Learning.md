# Laporan Proyek Machine Learning - Kornelius Setiawan

## Project Overview
Rekomendasi buku menjadi aspek penting dalam membantu pengguna menemukan bacaan yang sesuai dengan minat dan kebutuhannya. Dengan berkembangnya teknologi dan bertambahnya jumlah pengguna serta koleksi buku, sistem rekomendasi menjadi solusi efektif untuk meningkatkan pengalaman pengguna.

Proyek ini bertujuan untuk membangun sistem rekomendasi buku berbasis data yang dapat membantu pengguna memilih buku yang relevan berdasarkan preferensi mereka. Sistem rekomendasi ini dapat digunakan oleh berbagai platform penyedia buku online atau perpustakaan digital, hingga platform pendidikan untuk meningkatkan pengalaman pengguna dengan menyediakan buku yang sesuai dengan minat mereka.

Sistem rekomendasi dapat meningkatkan kepuasan pengguna dengan memberikan saran yang lebih personal dan relevan. Dalam konteks buku, ini membantu pengguna menemukan buku yang mungkin tidak mereka temui dalam pencarian biasa. Selain itu, sistem rekomendasi juga berpotensi meningkatkan penjualan atau penggunaan platform dengan meningkatkan keterlibatan pengguna.
  
[Book Recommendation System through content based and collaborative filtering method](https://ieeexplore.ieee.org/abstract/document/7684166) 
[Book Recommendation System](https://ieeexplore.ieee.org/abstract/document/9579647)
[College Library Personalized Recommendation System Based on Hybrid Recommendation Algorithm](https://www.sciencedirect.com/science/article/pii/S2212827119307401)

## Business Understanding
### Problem Statements
- Bagaimana cara membantu pengguna menemukan buku yang relevan dengan preferensi pengguna?
- Bagaimana cara memberikan rekomendasi buku yang relevan dan personal kepada pengguna berdasarkan interaksi pengguna dengan buku yang telah diberi rating?
- Bagaimana menangani *cold-start problem* ketika pengguna atau buku baru belum memiliki riwayat rating? 

### Goals
- Mengembangkan sistem rekomendasi yang dapat memberikan saran buku secara personal.
- Memberikan rekomendasi buku yang relevan berdasarkan rating yang diberikan oleh pengguna, menggunakan pendekatan Collaborative Filtering.
- Mengatasi cold-start problem dengan metode Content Based Filtering. 

### Solution statements
- Menerapkan **Content-Based Filtering** yang menggunakan informasi yang ada pada metadata buku, seperti penulis dan penerbit, untuk memberikan rekomendasi berdasarkan kesamaan buku. Pendekatan ini berguna ketika pengguna atau buku baru tidak memiliki riwayat rating.
- Menerapkan **Collaborative Filtering** yang menggunakan informasi rating dari pengguna untuk memberikan rekomendasi kepada pengguna lain yang memiliki pola rating yang serupa.

## Data Understanding
Dataset yang digunakan berisi 3 file yaitu, Books.csv, Rating.csv, dan User.csv. Dengan rincian sebagai berikut :
 - **Books.csv** memiliki 8 kolom dan 271360 baris, diantaranya :
    - ISBN: kode unik buku
    - Book-Title: judul buku
    - Book-Author: penulis buku
    - Year-Of-Publication: tahun terbit buku
    - Publisher: penerbit buku
    - Image-URL-S: URL gambar cover buku dengan ukuran S
    - Image-URL-M: URL gambar cover buku dengan ukuran M
    - Image-URL-L: URL gambar cover buku dengan ukuran L
    
    Kondisi dataset:
    - Ditemukan beberapa missing value pada dataset, khususnya pada kolom `Image-URL-L`.

 - **Rating.csv** memiliki 3 kolom dan 1149780 baris, diantaranya :
    - User-ID: ID unik pengguna yang memberi rating 
    - ISBN: ID buku yang diberi rating
    - Book-Rating: nilai rating buku yang diberikan pengguna (0-10) 

    Kondisi dataset:
    - Tidak ditemukan missing value pada dataset.

 - **Users.csv** memiliki 3 kolom dan 278858 baris, diantaranya : 
    - User-ID: ID unik pengguna
    - Location: lokasi pengguna
    - Age: usia pengguna

    Kondisi dataset:
    - Ditemukan beberapa missing value pada dataset, khususnya pada kolom `Age`.

Dataset yang digunakan berasal dari Kaggle : [Books Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

## Data Preparation
Untuk mempersiapkan data sebelum melakukan pemodelan, beberapa langkah perlu dilakukan:
### Merge Data
Langkah yang dilakukan dalam tahap ini yaitu menggabungkan Ratings dan Books berdasarkan ISBN.
```
data = pd.merge(rating, books, on='ISBN')
```
Penggabungan ini diperlukan agar data yang digunakan dalam model mencakup informasi lengkap mengenai rating yang diberikan, buku yang diberi rating, serta metadata buku seperti judul dan penulis.

### Drop Column yang tidak relevan dan Missing Values
Langkah yang dilakukan dalam tahap ini adalah menghapus beberapa kolom yang dianggap tidak relevan seperti `Image-URL` dan menghapus nilai yang hilang (missing values).
```
data_clean = data.drop(['Year-Of-Publication','Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1).dropna()
```
Pembersihan data dilakukan untuk menghindari pengaruh negatif dari kolom yang tidak relevan atau memiliki banyak nilai yang hilang. Data yang bersih dan relevan sangat penting untuk membangun model yang efektif.

### Melakukan normalisasi teks pada nama penulis untuk menghindari karakter non-ASCII.
Karakter non-ASCII dapat menyebabkan masalah saat pengolahan data dan analisis. Misalnya, jika nama penulis mengandung aksen (seperti "é" atau "ç"), kita dapat mengganti karakter-karakter tersebut dengan bentuk ASCII yang setara (misalnya, "é" menjadi "e" dan "ç" menjadi "c"). Langkah ini berutjuan untuk memastikan data yang digunakan bersih dan relevan.

Normalisasi nama penulis diperlukan agar teks yang digunakan dalam analisis bebas dari karakter non-ASCII yang dapat mengganggu proses pemodelan.

## Modeling
### Content-Based Filtering
Content-Based Filtering (CBF) adalah teknik rekomendasi yang memberikan rekomendasi berdasarkan konten yang terkait dengan item yang telah dipilih oleh pengguna. Dalam kasus ini, digunakan konten buku seperti Book-Author dan Publisher untuk menghasilkan rekomendasi buku yang mirip dengan buku yang telah diberi rating oleh pengguna.

Proses ini dilakukan dengan menggunakan TF-IDF Vectorizer untuk mengubah teks menjadi vektor numerik, lalu dihitung kesamaan antar vektor menggunakan cosine similarity. Cosine similarity digunakan untuk mengukur kesamaan antar buku berdasarkan fitur teks mereka. Buku yang memiliki kemiripan tertinggi dengan buku yang diberikan rating oleh pengguna akan direkomendasikan.

Parameter yang Digunakan : 

- `n-gram range: (1, 2)` : Menggunakan unigram (kata tunggal) dan bigram (kombinasi dua kata) untuk fitur teks.
- `min_df: 2` : Memastikan bahwa hanya kata-kata yang muncul lebih dari sekali yang digunakan dalam representasi.
- `stop_words: 'english'` : Mengabaikan kata-kata umum dalam bahasa Inggris (seperti "the", "and", dll.).

```
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
data_clean['content'] = data_clean['Book-Author'] + ' ' + data_clean['Publisher']

tf.fit(data_clean['content'])
```
Kelebihan : 
- Tidak membutuhkan data pengguna (cocok untuk cold start problem).
- Memberikan rekomendasi buku yang memiliki kesamaan tertinggi dengan buku yang dipilih pengguna.

Kekurangan :
- Tidak memberi variasi dalam rekomendasi, hanya berdasarkan konten yang serupa.

### Collaborative Filtering
Collaborative Filtering (CF) adalah teknik rekomendasi yang mengandalkan data interaksi pengguna dengan item untuk memberikan rekomendasi. Proyek ini menggunakan pendekatan berbasis Deep Learning dengan arsitektur sederhana bernama RecommenderNet. Model ini memetakan pengguna dan buku ke dalam embedding, lalu memprediksi rating berdasarkan interaksi dari embedding tersebut.

Parameter yang Digunakan : 

- `num_users` : Jumlah unik pengguna dalam dataset.
- `num_book` : Jumlah unik buku dalam dataset.
- `50`: Dimensi ruang embedding untuk pengguna dan buku. Dengan kata lain, ini adalah ukuran vektor representasi yang akan digunakan untuk mewakili pengguna dan buku dalam ruang vektor.


```
model = RecommenderNet(num_users, num_book, 50) # inisialisasi model

model.compile(
    loss= tf.keras.losses.MeanSquaredError(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```

Kelebihan : 
- Memberikan rekomendasi yang sangat personal berdasarkan interaksi pengguna.
- Dapat menangkap pola yang lebih dalam dan kompleks
 
Kekurangan : 
- Cold-start problem untuk pengguna atau buku baru.
- Waktu training lebih lama.

## Evaluation
Evaluasi dilakukan untuk menilai seberapa akurat sistem dalam memprediksi rating yang diberikan oleh pengguna terhadap buku. Dalam proyek ini, metrik yang digunakan adalah Root Mean Squared Error (RMSE). Metrik ini mengukur kesalahan antara rating yang diprediksi oleh model dengan rating aktual yang diberikan oleh pengguna.

Root Mean Squared Error adalah akar kuadrat dari rata-rata kuadrat kesalahan prediksi. Semakin kecil nilai RMSE, semakin baik performa model dalam memprediksi nilai rating yang sebenarnya.

**Hasil Evaluasi**

![Evaluasi](https://res.cloudinary.com/daoqz3rdr/image/upload/v1745582883/metric_l9lw4f.png)

Training RMSE: 0.2278
Validation RMSE: 0.3396

Nilai RMSE validasi yang relatif kecil menunjukkan bahwa model berhasil mempelajari pola dari data dengan cukup baik dan dapat memberikan prediksi rating yang akurat.

Model rekomendasi yang dikembangkan menggunakan dua pendekatan utama, yaitu `Content-Based Filtering` dan `Collaborative Filtering`. Kedua pendekatan ini berhasil menjawab masalah yang diidentifikasi dalam Problem Statement. Dengan `Content-Based Filtering`, sistem dapat memberikan rekomendasi buku yang relevan berdasarkan metadata buku, seperti penulis dan penerbit, yang sangat berguna dalam mengatasi masalah cold-start, yaitu ketika buku atau pengguna baru belum memiliki cukup data rating. Sedangkan, dengan `Collaborative Filtering`, sistem dapat memberikan rekomendasi yang lebih personal berdasarkan interaksi rating yang diberikan oleh pengguna lain yang memiliki preferensi serupa. Ini memastikan bahwa rekomendasi yang diberikan lebih sesuai dengan minat dan kebutuhan pengguna.

Berbagai Goals yang telah ditetapkan juga tercapai dengan baik. Model ini tidak hanya memberikan rekomendasi buku yang relevan dan personal berdasarkan rating pengguna, tetapi juga mengatasi masalah cold-start dengan menggunakan teknik `Content-Based Filtering` untuk memberikan rekomendasi meskipun data rating pengguna terbatas. Selain itu, `Collaborative Filtering` berfungsi dengan baik dalam memberikan rekomendasi berbasis pola rating yang serupa antar pengguna, meningkatkan relevansi rekomendasi bagi pengguna yang sudah lebih aktif dalam memberikan rating.

Secara keseluruhan, kedua solusi statement yang sudah direncanakan ini memiliki dampak yang besar terhadap kualitas sistem rekomendasi yang dibangun. `Content-Based Filtering` membantu menciptakan dasar rekomendasi yang relevan meskipun data pengguna terbatas, dan `Collaborative Filtering` meningkatkan pengalaman pengguna dengan memberikan rekomendasi yang lebih personal dan berdasarkan interaksi antar pengguna. Dengan demikian, kedua solusi ini telah memberikan kontribusi yang signifikan dalam mencapai tujuan untuk memberikan rekomendasi yang lebih baik dan lebih sesuai dengan preferensi pengguna.




