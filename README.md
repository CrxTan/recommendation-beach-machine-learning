Dokumentasi API Layanan Machine Learning PantaiKu
Selamat datang di dokumentasi API untuk Layanan Machine Learning (ML) PantaiKu. Layanan ini menyediakan fungsionalitas untuk analisis sentimen ulasan pantai dan rekomendasi pantai berdasarkan preferensi pengguna.

Base URL: http://localhost:5001 (Sesuaikan dengan URL server pengembangan Anda jika berbeda)

Daftar Isi
Endpoint Analisis Sentimen

Endpoint Rekomendasi Pantai

Skema Data Umum

Endpoint Analisis Sentimen

1. Analisis Sentimen Ulasan
   POST /analyze-sentiment

Menganalisis teks ulasan untuk menentukan sentimennya (Positif atau Negatif) beserta skor kepercayaan (confidence score). Endpoint ini melakukan preprocessing teks termasuk cleaning, case folding, perbaikan slang, tokenizing, filtering stopwords, dan stemming sebelum melakukan prediksi menggunakan model sentimen.

Request Body: application/json

{
"review_text": "Pantainya sangat indah dan bersih, saya suka sekali!"
}

review_text (string, required): Teks ulasan yang akan dianalisis.

Responses:

200 OK: Analisis sentimen berhasil.

{
"sentiment": "Positive", // atau "Negative"
"confidence": 0.9567 // Skor float antara 0.0 dan 1.0
}

400 Bad Request: Jika review_text tidak disertakan dalam request.

{
"error": "No review_text provided"
}

500 Internal Server Error: Jika model ML tidak berhasil dimuat atau terjadi kesalahan internal lainnya.

{
"error": "ML models not loaded or still initializing."
}

Endpoint Rekomendasi Pantai

1. Dapatkan Rekomendasi Pantai (Dummy)
   POST /recommend-beach

Memberikan rekomendasi pantai berdasarkan teks preferensi pengguna. PENTING: Implementasi saat ini menggunakan data dummy untuk rekomendasi.

Request Body: application/json

{
"preference_text": "Saya mencari pantai yang tenang, cocok untuk keluarga, dan memiliki fasilitas snorkeling."
}

preference_text (string, required): Teks yang mendeskripsikan preferensi pantai pengguna.

Responses:

200 OK: Rekomendasi (dummy) berhasil diterima.

{
"recommendations": [
{"placeId": "timur_0001", "score": 0.95},
{"placeId": "barat_0005", "score": 0.88},
{"placeId": "tengah_0010", "score": 0.76}
]
}

placeId (string): ID unik untuk pantai yang direkomendasikan.

score (float): Skor relevansi rekomendasi (semakin tinggi semakin baik).

400 Bad Request: Jika preference_text tidak disertakan dalam request.

{
"error": "No preference_text provided"
}

Skema Data Umum
Skema Error Umum
Response error umumnya mengikuti format ini:

{
"error": "Deskripsi error yang relevan"
}

Catatan Tambahan
Preprocessing Teks: Endpoint /analyze-sentiment melakukan langkah-langkah preprocessing berikut pada review_text sebelum analisis:

Cleaning: Menghapus mention (@), hashtag (#), RT, URL, angka, dan karakter non-alfabet.

Case Folding: Mengubah semua teks menjadi huruf kecil.

Fix Slangwords: Mengganti kata-kata slang dengan bentuk bakunya (berdasarkan kamus slangwords yang ada di kode).

Tokenizing: Memecah teks menjadi token (kata).

Filtering (Stopword Removal): Menghapus stopwords (berdasarkan daftar all_stopwords).

Stemming: Mengubah kata ke bentuk dasarnya menggunakan Sastrawi Stemmer.

Model dan Aset: Layanan ini bergantung pada model sentimen (.h5), tokenizer (.json), dan lexicon positif/negatif (.json) yang harus tersedia di path yang benar saat aplikasi dimulai.

MAX_SEQUENCE_LENGTH: Untuk endpoint /analyze-sentiment, MAX_SEQUENCE_LENGTH diatur ke 10. Ini berarti sequence input akan di-padding atau di-truncate menjadi panjang 10 sebelum dimasukkan ke model. Sesuaikan nilai ini jika model Anda dilatih dengan panjang sequence yang berbeda.

CORS: Cross-Origin Resource Sharing (CORS) diaktifkan untuk semua rute, memungkinkan request dari domain yang berbeda.
