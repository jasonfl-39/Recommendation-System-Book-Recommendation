# Laporan Proyek Machine Learning - Jason Filbert Leo

## Project Overview

Pada suatu website vendor buku, sistem rekomendasi dapat digunakan untuk menyarankan buku tertentu kepada pengguna sehingga dapat mengarahkan pengguna ke jenis buku yang diinginkan. Terdapat beberapa cara untuk membangun sistem ini, antara lain berdasarkan pada konten item, collaborative filtering, dan association mining. (Parvatikar & Joshi, 2015)

Dalam proyek ini, akan dibangun suatu sistem rekomendasi berdasarkan collaborative filtering untuk memberikan 10 rekomendasi buku terbaik ke user dengan menggunakan data rating yang telah diberikan oleh user sebelumnya. Model akan ditraining dan dievaluasi sehingga diperoleh sistem rekomendasi yang dapat memberikan saran buku secara akurat.

Referensi: [S. Parvatikar and B. Joshi, "Online book recommendation system by using collaborative filtering and association mining," 2015 IEEE International Conference on Computational Intelligence and Computing Research (ICCIC), Madurai, India, 2015, pp. 1-4, doi: 10.1109/ICCIC.2015.7435717.](https://ieeexplore.ieee.org/abstract/document/7435717)

## Business Understanding

Untuk mendalami masalah sistem rekomendasi yang akan dibahas pada proyek ini, maka dibahas terlebih dahulu pernyataan masalah dan tujuan yang ingin dicapai dari proyek ini.

### Problem Statements

Pernyataan masalah:
- Bagaimana cara untuk memberikan rekomendasi buku kepada user berdasarkan pada data rating user?
- Apakah memungkinkan untuk memberikan rekomendasi buku berupa buku yang belum dilihat oleh user saja, tanpa memengaruhi model saat memproses ?

### Goals

Tujuan proyek:
- Untuk membentuk suatu sistem rekomendasi berdasarkan pada data rating, dapat digunakan algoritma Collaborative Filtering. Pada algoritma ini, data rating user akan diproses sehingga diperoleh bias user dan bias buku yang kemudian menjadi dasar model untuk menentukan buku mana saja yang paling baik untuk direkomendasikan ke setiap user.
- Jika dalam data rating tersebut ikut ditracking pula saat user tidak memberikan rating secara eksplisit kepada suatu buku walaupun sudah melihat buku tersebut maka memungkinkan untuk menyesuaikan model sehingga hanya memberikan buku-buku yang belum dilihat saja sebagai output. Sedangkan model hanya menggunakan data rating eksplisit sebagai patokan untuk memberikan rekomendasi, dapat ditambahkan variabel baru dengan menggunakan keseluruhan data untuk mengecek jika suatu buku sudah dilihat oleh user atau belum.

## Data Understanding
Pada proyek ini, digunakan dataset rekomendasi buku dari Kaggle: [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?)

### Komposisi File beserta Variabel pada Dataset
Terdapat tiga file csv pada dataset ini, yaitu:
1. Books.csv: Database buku, terdiri atas 271.360 sampel. Pada file ini terdapat variabel-variabel berikut:
    - `ISBN`: Nomor ISBN dari setiap buku pada database.
    - `Book-Title`: Judul dari setiap buku pada database.
    - `Book-Author`: Penulis dari setiap buku pada database.
    - `Year-Of-Publication`: Tahun terbit dari setiap buku pada database.
    - `Publisher`: Penerbit dari setiap buku pada database.
    - `Image-URL-S`: Gambar cover buku, ukuran kecil.
    - `Image-URL-M`: Gambar cover buku, ukuran sedang.
    - `Image-URL-L`: Gambar cover buku, ukuran besar.
2. Users.csv: Database pengguna, terdiri atas 278.858 sampel. Pada file ini terdapat variabel-variabel berikut:
    - `User-ID`: Nomor ID dari setiap user pada database.
    - `Location`: Lokasi tempat tinggal setiap user pada database.
    - `Age`: Usia dari setiap user pada database.
3. Ratings.csv: Database rating buku oleh pengguna, terdiri atas 1.149.780 sampel. Pada file ini terdapat variabel-variabel berikut:
    - `User-ID`: Nomor ID dari setiap user pada database.
    - `ISBN`: Nomor ISBN dari setiap buku pada database.
    - `Book-Rating`: Nilai rating buku yang diberikan oleh user. Dapat bernilai 1-10 sebagai rating eksplisit, atau 0 sebagai rating implisit di mana user telah melihat buku tetapi tidak memberikan rating

### Exploratory Data Analysis
Mengingat bahwa penggunaan variabel dengan tanda '-' dapat memicu masalah pada Python, sebelum diproses lebih lanjut semua variabel dengan tanda '-' diubah namanya sehinga tidak mengandung tanda '-' lagi. Selain itu, karena gambar cover buku tidak diperlukan dalam proyek ini maka variabel `Image-URL-S`, `Image-URL-M`, dan `Image-URL-L` telah didrop dahulu dari dataframe setelah loading data.

Selanjutnya dapat diperiksa jenis data pada setiap file:
Pada dataframe book\_df yang berisi dari Books.csv, diperoleh 271.360 entri dengan dua entri kosong pada kolom Book\_Author dan Publisher.

![Naamloos3](https://github.com/user-attachments/assets/e46fe23c-aa79-47c1-8785-e6b785ba45c8)

Pada dataframe user_df yang berisi dari Users.csv, diperoleh 278.858 entri dengan jumlah entri kosong cukup signifikan pada kolom Age.

![Naamloos5](https://github.com/user-attachments/assets/338cd5fc-9d1e-4d9b-b967-756fa54af482)

Pada dataframe rating_df yang berisi dari Ratings.csv, diperoleh 1.149.780 entri tanpa missing value.

![Naamloos7](https://github.com/user-attachments/assets/58e9b420-3d4c-4927-b8b1-0c2b4e2a0619)

Selanjutnya, diperoleh pula jumlah entri data unik pada book\_df dan user\_df:

![image](https://github.com/user-attachments/assets/1f67f778-4310-4121-9547-46e5dd5ebef5)

Karena jumlahnya sesuai dengan jumlah entri, maka tidak ada data duplikat pada book\_df ataupun user\_df.
Diperhatikan bahwa fitur `User_ID` sudah berbentuk integer dan tidak ada missing value ataupun duplikat sehingga dapat digunakan pada algoritma Collaborative Filtering tanpa encoding nantinya. Sementara itu fitur `ISBN` berbentuk object sehingga akan diperlukan encoding supaya dapat digunakan. Pada model sendiri, fitur yang akan diperlukan hanya fitur `User_ID`, `ISBN`, `Book-Title`, dan `Book-Rating`, sehingga di luar beberapa penyesuaian untuk memastikan tidak ada entri tidak valid pada data, fitur-fitur lainnya dapat diabaikan pada proyek ini. 

Pada book\_df, diperoleh jumlah data unik untuk setiap fitur berikut:

![Naamloos4](https://github.com/user-attachments/assets/079cb962-36a1-4631-8957-995beacdc74e)

Pada rating\_df, diperoleh jumlah data unik untuk setiap fitur berikut:

![Naamloos8](https://github.com/user-attachments/assets/b6d88899-bbb3-46a4-bd0b-2651b7162239)

Sebelumnya telah dibahas bahwa pada data rating terdapat jenis rating eksplisit di mana user memberikan nilai rating 1-10 kepada suatu buku, atau rating implisit bernilai 0 di mana user telah melihat buku tetapi tidak memberikan nilai rating. Proses training model hanya akan mengambil data eksplisit pada supaya nilai rating 0 tidak membuat suatu buku tertentu seakan-akan tidak disukai oleh user padahal user hanya tidak memberikan nilai tertentu yang belum tentu berarti bahwa user tidak menyukai buku tersebut. Sementara itu, keseluruhan data buku, baik eksplisit dan implisit, tetap akan ditracking sehingga sistem rekomendasi hanya akan menyarankan buku yang belum dilihat user. Karena itu, dilakukan pembagian data rating menjadi data rating eksplisit dan implisit. Diperoleh jumlah data berikut:

![Naamloos9](https://github.com/user-attachments/assets/3088806d-3e60-4b3d-a7cf-94a0102e23ad)

dengan jumlah data user pada setiap pembagian data rating:

![image](https://github.com/user-attachments/assets/f2fd28b5-3194-4a9d-9be4-9c734f87986b)

dan jumlah data buku pada setiap pembagian data rating:

![image](https://github.com/user-attachments/assets/59b2b11a-8671-4c67-b223-d5f97b0e4a07)

Selanjutnya, data buku dipasangkan dengan data rating keseluruhan yang akan digunakan oleh variabel yang akan mentracking buku apa saja yang belum dibaca user. Diperoleh data berikut:

![image](https://github.com/user-attachments/assets/e25b89b8-4ace-4632-8585-8a0fe823ce21)

Data hasil pemasangan ini dimasukkan dalam variabel `all_books_clean`, di mana diperoleh sejumlah missing value berikut:

![image](https://github.com/user-attachments/assets/097a323e-7a0f-4ad7-bbee-1b0ccda4fbb9)

Data buku juga  dipasangkan dengan data rating eksplisit yang akan dipakai pada proses training. Diperoleh data berikut:

![image](https://github.com/user-attachments/assets/d77967a8-4937-4df8-a0ab-1ddbab7ccd74)

Data hasil pemasangan ini dimasukkan dalam variabel `exp_books_clean`, di mana diperoleh sejumlah missing value berikut:

![Naamloos19](https://github.com/user-attachments/assets/537479f6-6a12-48ba-9750-a058fb4b2171)

## Data Preparation
Sebelum data dapat digunakan pada model, ada beberapa tahapan pembersihan data yang harus dilakukan terlebih dahulu.

### Mengatasi Missing Value
Pada Exploratory Data Analysis, terlihat bahwa terdapat beberapa missing value dari hasil pemasangan data rating dengan data buku, baik pada data keseluruhan dan data eksplisit.
Pada data keseluruhan, sebelumnya terdapat 118.650 missing value
Snippet kode:
```
all_books_clean = all_books_title.dropna()
all_books_clean.isnull().sum()
```
Diperoleh bahwa tidak ada missing value lagi pada data keseluruhan.

![Naamloos20](https://github.com/user-attachments/assets/66d2b5e0-a600-4881-9480-477d032cb35c)

Pada data eksplisit, sebelumnya terdapat 49.832 missing value
Snippet kode:
```
exp_books_clean = exp_books_title.dropna()
exp_books_clean.isnull().sum()
```
Diperoleh bahwa tidak ada missing value lagi pada data eksplisit.

![Naamloos20](https://github.com/user-attachments/assets/66d2b5e0-a600-4881-9480-477d032cb35c)

### Encoding ISBN
Sebelumnya, terlihat bahwa fitur ISBN memiliki dtype object yang tidak bisa diproses secara langsung oleh model. Oleh karena itu perlu dilakukan encoding data ini menjadi integer sehingga dapat digunakan.

Snippet kode:
```
isbns = exp_books_clean['ISBN'].unique().tolist()
print('list ISBN: ', isbns)
 
book_isbn_encoded = {x: i for i, x in enumerate(isbns)}
print('encoded ISBN : ', book_isbn_encoded)
 
book_encoded_to_isbn = {i: x for i, x in enumerate(isbns)}
```
Hal yang sama tidak perlu dilakukan pada User ID karena sudah bersih dan berbentuk integer.

### Sorting Data Buku hingga menjadi Dictionary
Sebelum membentuk variabel untuk mentracking buku apa saja yang telah dilihat user sehingga tidak ikut serta direkomendasikan, diperlukan adanya dictionary berisi pasangan ISBN dan judul buku untuk dibandingkan dengan buku yang telah dilihat oleh user.
Dari pasangan data rating dan buku `all_books_clean`yang telah disorting dan disimpan dalam variabel baru `book_sort`, diperoleh 383.839 baris data di mana hanya terdapat 149833 nilai ISBN unik, yang menunjukkan bahwa terdapat banyak data duplikat yang harus didrop.

Snippet kode:
```
book_sort = book_sort.drop_duplicates('ISBN')
book_sort
```
Data sebelum duplikat didrop:

![Naamloos16](https://github.com/user-attachments/assets/34d6d773-be4a-4af8-bc5f-302c3f511d3a)

Setelah data duplikat didrop:

![Naamloos17](https://github.com/user-attachments/assets/9a3eeb9a-c54f-491d-ba20-607443509cee)

Sudah diperoleh 149.833 baris data dengan jumlah ISBN dan judul sama.
Selanjutnya daftar setiap ISBN dan judul dikumpulkan dalam list, dan kemudian dimasukkan ke dalam dictionary `book_dict`.

Snippet kode:
```
isbn = book_sort['ISBN'].tolist()
title = book_sort['Book_Title'].tolist()

book_dict = pd.DataFrame({
    'isbn': isbn,
    'title': title,
})
book_dict
```
Diperoleh dictionary berikut:

![Naamloos18](https://github.com/user-attachments/assets/1c419908-1cdd-4b66-906c-be5100af83a4)

### Train-Test Split
Untuk melakukan validasi model, dataset yang telah dipasangkan dan dibersihkan perlu dibagi menjadi training set dan testing set. Berhubungan dengan ukuran data yang cukup besar, diambil rasio train:test pada 90:10. Sebelum pembagian data, pada variabel x User ID dan encoding ISBN dicocokan menjadi satu value. Pada variabel y, nilai rating dinormalisasi menjadi berada pada range 0-1.

Snippet kode:
```
x = df[['User_ID', 'book']].values
y = df['Book_Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
 
train_indices = int(0.9 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
```

## Modeling
Untuk memproses dataset ini sehingga diperoleh suatu sistem rekomendasi, telah dipilih untuk menggunakan algoritma Collaborative Filtering. Alasannya adalah karena pada dataset terdapat data rating buku dari user berjumlah besar yang cocok diaplikasikan pada algoritma ini. Hal ini tidak hanya membuat model dapat mengetahui tentang nilai rating user dari 1-10, tetapi juga menelusuri buku apa saja yang telah dilihat oleh user. Jadi, berdasarkan pada data ini dapat diprediksi buku apa yang belum dilihat oleh user yang akan menarik perhatian user untuk membaca buku tersebut berdasarkan pada rating yang telah diberikan sebelumnya.
Sementara itu, jumlah data yang sangat besar disertai dengan kurangnya informasi tentang konten atau genre setiap buku membuat algoritma Content Based Filtering kurang memadai untuk digunakan pada dataset ini karena sulitnya mengekstrak fitur dengan TF-IDF Vectorizer dan Cosine Similarity dari kolom-kolom pada data buku yang tersedia (judul buku, penulis, dan perusahaan pencetak saja). Karena itu, algoritma tersebut tidak digunakan pada proyek ini.

Snippet kode:
```
import tensorflow as tf
class RecommenderNet(tf.keras.Model):
 
  # Insialisasi fungsi
  def __init__(self, num_user, num_book, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_user = num_user
    self.num_book = num_book
    self.embedding_size = embedding_size
    self.user_embedding = tf.keras.layers.Embedding( 
        num_user,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = tf.keras.regularizers.l2(1e-6)
    )
    self.user_bias = tf.keras.layers.Embedding(num_user, 1) 
    self.book_embedding = tf.keras.layers.Embedding( 
        num_book,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = tf.keras.regularizers.l2(1e-6)
    )
    self.book_bias = tf.keras.layers.Embedding(num_book, 1)
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0])
    user_bias = self.user_bias(inputs[:, 0])
    book_vector = self.book_embedding(inputs[:, 1]) 
    book_bias = self.book_bias(inputs[:, 1])
 
    dot_user_book = tf.tensordot(user_vector, book_vector, 2) 
 
    x = dot_user_book + user_bias + book_bias
    
    return tf.nn.sigmoid(x)
```
Untuk algoritma Collaborative Filtering yang digunakan pada modeling, secara umum algoritma ini bekerja dengan cara menganalisis preferensi setiap user berdasarkan pada data rating. Jika User A dan User B menyukai beberapa buku yang sama, maka model dengan algoritma ini akan memberikan rekomendasi kepada User A berupa buku-buku lainnya yang disukai User B tetapi belum dilihat oleh User A. Hal ini diukur dengan perhitungan vektor dan bias dari user dan buku di mana hasilnya kemudian akan diproses oleh fungsi sigmoid menghasilkan skor kecocokan seorang user dengan buku tertentu pada range 0-1. Buku dengan skor kecocokan terbaik akan dipilih sebagai rekomendasi buku oleh model.
Selain itu, karena data yang digunakan pada model ini meliputi rating eksplisit yang diberikan oleh user dan rating implisit di mana user telah melihat suatu buku tanpa memberikan nilai rating, maka model juga akan mengecek buku mana saja yang belum pernah dilihat user untuk direkomendasikan dengan variabel `book_not_seen` yang dapat dilihat pada snippet kode ketiga di bagian Evaluation. Dari model ini, hasil output akan diiterasikan sehingga diperoleh 10 rekomendasi terbaik kepada user.

## Evaluation
Untuk proses training pada model, digunakan fungsi loss Binary Crossentropy untuk mengevaluasi tingkat error fungsi sigmoid sebagai output dari model, fungsi optimizer model dengan fungsi Adam menggunakan parameter learning_rate = 0.0001, dan diambil metrik RMSE (Root Mean Squared Error) untuk evaluasi. Metrik ini merupakan nilai akar dari error Mean Squared Error (MSE) dan berfungsi mengukur tingkat akurasi prediksi suatu model pada dataset tertentu, di mana dalam masalah ini merupakan prediksi rekomendasi buku yang diberikan berdasarkan data rating keseluruhan.

Snippet kode:
```
model = RecommenderNet(num_user, num_book, 50) # inisialisasi model
 
# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```
Proses training dilakukan dengan mengambil ukuran batch = 512  dan jumlah epoch = 10.
Snippet kode:
```
history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 512,
    epochs = 10,
    validation_data = (x_val, y_val)
)
```
Menurut hasil dari training, diperoleh grafik berikut:

![download](https://github.com/user-attachments/assets/0a6a3946-6460-473e-9c73-94eff1925ebf)

Diperoleh tingkat error secara umum menurun pada setiap epoch, baik pada training dan validation set. Ada kemungkinan bahwa model masih underfit tetapi mengingat ukuran dataset yang sangat besar, akan memakan waktu dan beban komputasi yang sangat besar jika diambil ukuran batch lebih kecil dan jumlah epoch lebih banyak.

Snippet kode:
```
ratings = model.predict(user_book_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_book_ids = [
    book_encoded_to_isbn.get(book_not_seen[x][0]) for x in top_ratings_indices
]
 
print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Book with high ratings from user')
print('----' * 8)
 
top_book_user = (
    book_seen.sort_values(
        by = 'Book_Rating',
        ascending=False
    )
    .head(5)
    .ISBN.values
)
 
book_df_rows = book_df[book_df['isbn'].isin(top_book_user)]
for row in book_df_rows.itertuples():
    print(row.isbn, ':', row.title)
 
print('----' * 8)
print('Top 10 book recommendation')
print('----' * 8)
 
recommended_book = book_df[book_df['isbn'].isin(recommended_book_ids)]
for row in recommended_book.itertuples():
    print(row.isbn, ':', row.title)
```
Hasil dari model ini pada salah satu user random memberikan output berikut:

![Naamloos1](https://github.com/user-attachments/assets/f844f6eb-3688-47e1-a5a3-94b73efbb455)

Diperoleh bahwa sistem rekomendasi berhasil memberikan rekomendasi 10 buku relevan yang mungkin akan disukai user berdasarkan pada rating eksplisit dari user dan buku lainnya yang telah dilihat oleh user. Dapat disimpulkan bahwa sistem rekomendasi ini sudah berjalan dengan cukup baik dalam menyelesaikan masalah-masalah pada perihal sistem rekomendasi buku kepada user seperti tertera pada Business Understanding. Untuk kelanjutannya, sistem ini masih bisa disempurnakan lagi jika terdapat waktu dan kemampuan komputasi lebih tinggi untuk melakukan training sehingga bisa menjadi lebih akurat lagi dalam memberikan rekomendasi.
