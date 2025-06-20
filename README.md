# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding

**Jaya Jaya Institut** merupakan salah satu institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun 2000. Hingga saat ini, institut ini telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat cukup banyak siswa yang tidak menyelesaikan pendidikannya alias **dropout**.

Tingginya jumlah dropout menjadi masalah serius bagi institusi pendidikan karena dapat memengaruhi reputasi, efisiensi operasional, serta keberhasilan proses belajar-mengajar. Oleh karena itu, pihak Jaya Jaya Institut ingin mendeteksi **lebih awal** siswa yang berpotensi dropout agar dapat diberikan **bimbingan khusus**.

Sebagai calon data scientist masa depan, kita diminta untuk membantu Jaya Jaya Institut menyelesaikan masalah ini dengan menggunakan pendekatan **machine learning dan dashboard analitik**.

### Permasalahan Bisnis

Jaya Jaya Institut menghadapi tantangan serius dalam hal tingginya angka mahasiswa dropout yang berpotensi mengganggu reputasi akademik, akreditasi, serta efisiensi proses pembelajaran. Institusi kesulitan dalam:
- Mengidentifikasi faktor utama yang mendorong mahasiswa untuk berhenti studi di tengah jalan.
- Tidak adanya sistem prediksi berbasis data yang dapat memberikan peringatan dini bagi mahasiswa yang berisiko tinggi dropout.
- Kesulitan dalam monitoring performa akademik mahasiswa secara real-time dan menyeluruh dari sisi akademik dan sosial.
Oleh karena itu, institusi memerlukan pendekatan analitik dan sistem machine learning untuk membantu pengambilan keputusan yang berbasis data.

### Cakupan Proyek

- Mengidentifikasi faktor-faktor yang memengaruhi status mahasiswa (Graduate, Dropout, Enrolled)
- Membangun model prediksi status mahasiswa menggunakan supervised machine learning
- Menyusun dashboard analitik menggunakan Looker Studio untuk memvisualisasikan insight penting dari data
- Memberikan rekomendasi berbasis data untuk pengambilan keputusan akademik

---

## Persiapan

**Sumber data:**  
- Dataset asli: 
https://drive.google.com/file/d/1fXxRxV11Oa7lqhWHRralQx5iXMNyDk5N/view?usp=sharing
- Dataset bersih dan delimiter diperbaiki: 
https://drive.google.com/file/d/1kDE2zUzRWjqnP5pRK4l3yOJI_cTN-_Ua/view?usp=sharing

### Library Utama yang Digunakan
- `pandas`, `numpy`, `scikit-learn`, `streamlit`

---

### Setup Environment

Terdapat beberapa cara untuk menyiapkan lingkungan kerja proyek machine learning ini agar dapat dijalankan dengan lancar:

#### 🔹 Setup Environment - Anaconda
```bash
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt
```
### Setup Environment - Pipenv (Shell/Terminal)
```bash
pip install pipenv
pipenv install
pipenv shell
pip install -r requirements.txt
```

### Membuat dan Mengaktifkan Virtual Environment (venv)
Buka terminal

# Membuat virtual environment
python -m venv venv

# Mengaktifkan (Windows)
venv\Scripts\activate

# Mengaktifkan (Mac/Linux)
source venv/bin/activate

# Menginstal Dependensi dari requirements.txt
pip install -r requirements.txt

## Data Understanding
Mengenali isi deskripsi data, nilai unik status, jumlah data missing, dan tahap EDA serta visualisasi data.

**Insight:**  
- Dataset terdiri dari total 4425 baris, di mana 1 baris pertama merupakan header kolom dan 4424 sisanya adalah data mahasiswa.

- Informasi yang dikandung meliputi aspek akademik dan sosial, seperti nilai saat masuk, status beasiswa, kondisi pembayaran biaya kuliah, serta performa akademik per semester.

- Meskipun tidak ditemukan nilai kosong (NaN), proses pembersihan tetap dilakukan untuk menjaga konsistensi data.

- Variabel target Status menunjukkan distribusi yang tidak seimbang, dengan mayoritas mahasiswa berada di kelas Graduate, diikuti Dropout, dan paling sedikit di kelas Enrolled.

- Fitur seperti Curricular_units_1st_sem_approved, Curricular_units_1st_sem_grade, Curricular_units_2nd_sem_approved, Curricular_units_2nd_sem_grade, serta Tuition_fees_up_to_date menunjukkan distribusi yang berbeda-beda di setiap kelas dan berpotensi menjadi indikator penting dalam prediksi status mahasiswa.

- Dominasi kelas Graduate menandakan bahwa sebagian besar mahasiswa berhasil menyelesaikan studi, sementara dropout menempati proporsi menengah, dan mahasiswa aktif (Enrolled) menjadi kelompok minoritas.


## Data Preparation
- Memetakan label `status` menjadi `'dropout: 0'`, `'enrolled: 1'`, dan `'graduate:2'`
- Dilakukan encoding kolom kategorikal dan normalisasi fitur numerik saat proses modeling di notebook.

**Insight:**  
- Data kategorikal diubah menjadi numerik agar kompatibel dengan algoritma machine learning
Fitur seperti Status, Scholarship_holder, dan Debtor dikonversi ke format numerik menggunakan teknik encoding, sehingga dapat diproses langsung oleh model prediktif tanpa kehilangan informasi kategorikal.

- Standarisasi dilakukan untuk menyamakan skala antar fitur numerik
Kolom seperti Admission_grade, Age_at_enrollment, dan nilai akademik tiap semester memiliki rentang berbeda. Dengan standardisasi, proses pelatihan model menjadi lebih stabil dan hasil prediksi lebih akurat, terutama untuk model seperti SVM dan KNN yang sensitif terhadap skala.

- Data dibagi menjadi train dan test set dengan proporsi 80:20 untuk validasi yang adil
Penggunaan random_state=42 menjamin hasil pembagian data konsisten di setiap eksperimen, sehingga mempermudah reproduksi dan evaluasi model secara objektif.

## Modeling

Model yang digunakan:
- Logistic Regression
- Random Forest
- KNN
- SVM
- Gradient Boosting
- AdaBoost
- Extra Trees


**Insight:**  
- Tujuh algoritma machine learning digunakan untuk membandingkan performa prediksi
Model yang diuji mencakup pendekatan linear, distance-based, hingga ensemble: Logistic Regression, Random Forest, KNN, SVM, Gradient Boosting, AdaBoost, dan Extra Trees. Tujuannya untuk menemukan model yang paling optimal terhadap pola data.

- Gradient Boosting dan Logistic Regression tampil sebagai model terbaik dengan akurasi tertinggi
Kedua model ini konsisten mencetak akurasi antara 86% hingga 88%, menunjukkan kemampuannya dalam menangkap hubungan kompleks antara fitur dengan label dropout.

- Semua model menunjukkan performa yang baik dengan akurasi di atas 80%
Ini menandakan bahwa dataset memiliki informasi yang cukup kuat untuk membedakan status mahasiswa, dan preprocessing yang dilakukan mampu meningkatkan kualitas input ke model.

## Evaluation

Dilakukan evaluasi dengan:
- Confusion matrix
- Classification report
- Feature importance
- Precision-Recall curve

**Insight:**  
- Model memiliki performa sangat baik dalam mengenali kelas Dropout dan Graduate
Precision-Recall Curve menunjukkan bahwa kelas 0 (Dropout) dan kelas 2 (Graduate) memiliki average precision di atas 0.85, dengan kurva yang mulus dan stabil. Ini menunjukkan kemampuan model dalam mendeteksi dua kelas mayoritas secara konsisten.

- Performa rendah pada kelas Enrolled disebabkan oleh ketidakseimbangan jumlah data
Kelas 1 (Enrolled) hanya mencapai average precision sebesar 0.50, dengan kurva yang menurun tajam. Evaluasi melalui confusion matrix menunjukkan bahwa prediksi untuk kelas ini sering tertukar, mengindikasikan bahwa data kelas Enrolled jauh lebih sedikit dibandingkan kelas lain.

- Ketidakseimbangan kelas berdampak signifikan terhadap hasil evaluasi
Walaupun akurasi keseluruhan tinggi, performa model terhadap kelas minoritas seperti Enrolled perlu ditingkatkan menggunakan strategi seperti resampling, penyesuaian threshold, atau penambahan bobot kelas.

## Business Dashboard

Dashboard interaktif dibangun menggunakan Google Looker Studio, yang menampilkan insight visual berdasarkan data akademik mahasiswa. Tujuannya adalah agar pihak kampus dapat memahami tren dropout secara lebih intuitif. Visualisasi mencakup:
### Bar Chart Horizontal
Menampilkan total nilai Admission_grade per kategori status (Graduate, Dropout, Enrolled). Terlihat jelas bahwa mahasiswa yang lulus (Graduate) memiliki akumulasi nilai masuk tertinggi, sedangkan yang Enrolled memiliki jumlah yang jauh lebih sedikit.

### Pie Chart Komposisi Status Mahasiswa
Pie chart menunjukkan proporsi data per kelas. Komposisi ini mengonfirmasi ketidakseimbangan kelas yang ditemukan di tahap EDA, dengan mayoritas mahasiswa berhasil lulus.

### Scatter Plot
Visualisasi hubungan antara Admission_grade dan Curricular_units_2nd_sem_enrolled memperlihatkan persebaran mahasiswa. Ini membantu memahami apakah nilai masuk berkorelasi dengan jumlah mata kuliah yang diambil di semester lanjutan.

https://lookerstudio.google.com/reporting/15e1d75b-4b77-413a-8d74-b5f96efb9bdc

## Menjalankan Sistem Machine Learning
Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. Selain itu, sertakan juga link untuk mengakses prototype tersebut.
### Menjalankan aplikasi streamlit secara lokal
- Buka terminal
- run code : streamlit run App.py

Jika ingin membuka tidak secara lokal, dapat diakses melalui link dibawah ini:
https://prediksidrop.streamlit.app

## Conclusion

Berdasarkan hasil eksplorasi data dan modeling, ditemukan bahwa mahasiswa yang berpotensi mengalami **dropout** memiliki beberapa **karakteristik utama**, yaitu:

- Memiliki **jumlah mata kuliah yang disetujui (approved)** pada semester 1 dan 2 yang relatif rendah.
- Memiliki **nilai akademik semester 1 dan 2** yang cenderung di bawah rata-rata.
- **Status pembayaran biaya kuliah** yang tidak lancar (Tuition_fees_up_to_date = 0).
- Tidak menerima beasiswa (Scholarship_holder = 0).
- Beberapa mahasiswa dropout juga memiliki status sebagai **debtor** (penunggak pembayaran).

Dari sisi modeling, model **Gradient Boosting** dan **Logistic Regression** menunjukkan performa prediksi tertinggi (akurasi 86–88%). Namun, **Extra Trees Classifier** memberikan **feature importance** paling informatif dalam mengidentifikasi faktor-faktor kunci terhadap status mahasiswa.

Kelas **Dropout** dan **Graduate** dapat diprediksi dengan baik (average precision > 0.85), tetapi prediksi terhadap kelas **Enrolled** masih rendah akibat **ketimpangan distribusi kelas**.

Dengan mengintegrasikan sistem prediktif ini ke dalam sistem akademik, pihak kampus dapat:

- Melakukan **intervensi dini** kepada mahasiswa dengan risiko tinggi dropout.
- Menyusun **program bimbingan akademik** berbasis data.
- Menyediakan **dukungan finansial atau kebijakan pembayaran** yang lebih fleksibel.

Sistem ini dapat menjadi alat bantu penting dalam menurunkan angka dropout dan meningkatkan kualitas kelulusan secara berkelanjutan.

---

## Rekomendasi Action Items

1. **Intervensi Akademik Dini**
   - Fokus pada mahasiswa dengan nilai rendah dan sedikit mata kuliah yang disetujui pada semester awal.
   - Terapkan program remedial, bimbingan belajar, atau sistem alert akademik berbasis fitur: 
     - `Curricular_units_1st_sem_approved`
     - `Curricular_units_2nd_sem_approved`
     - `Curricular_units_1st/2nd_sem_grade`

2. **Kebijakan Pembayaran Adaptif**
   - Berdasarkan fitur `Tuition_fees_up_to_date`, mahasiswa yang tidak update pembayaran memiliki risiko tinggi untuk dropout.
   - Kampus dapat mengembangkan skema pembayaran fleksibel seperti cicilan atau penundaan dengan sistem monitoring otomatis.

3. **Penguatan Program Beasiswa**
   - Mahasiswa yang tidak mendapatkan beasiswa (`Scholarship_holder = 0`) lebih rentan terhadap dropout.
   - Tingkatkan sosialisasi dan akses program bantuan finansial kepada mahasiswa dengan performa akademik baik tetapi memiliki kendala ekonomi.

4. **Pemetaan Risiko Dropout Secara Visual**
   - Gunakan dashboard analitik untuk mengamati tren dropout secara instan dan menyeluruh, guna mendukung pengambilan keputusan cepat dan tepat.

---

