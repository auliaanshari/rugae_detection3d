# Panduan Menjalankan Pipeline Training Model Recognition (Triplet Loss)

Dokumen ini merinci langkah-langkah yang diperlukan untuk melatih dan mengevaluasi model *recognition* dari awal hingga akhir. Panduan ini ditujukan untuk anggota tim developer.

---
## 📦 Prasyarat

Sebelum memulai, pastikan hal-hal berikut sudah disiapkan:

1.  **Environment Python:** Pastikan Anda sudah memiliki *environment* `venv` atau `conda` yang aktif.
2.  **Dependensi:** Instal semua *library* yang dibutuhkan. Daftar di bawah ini bisa disimpan sebagai `requirements.txt` dan diinstal dengan `pip install -r requirements.txt`.
    ```txt
    torch
    torchvision
    torchaudio
    numpy
    open3d
    scikit-image
    pandas
    tqdm
    matplotlib
    seaborn
    scikit-learn
    ```
3.  **Hardware:** Sangat disarankan menggunakan **GPU NVIDIA dengan CUDA** yang sudah terinstal dan kompatibel dengan PyTorch untuk mempercepat proses training.
4.  **Data Anotasi:** Pastikan folder `annotated_dataset/` sudah berisi hasil kerja lengkap dari tim anotator (file `..._annotated.ply` dan `..._landmarks.txt` untuk setiap pasien).

---
## 📁 Struktur Folder

Pipeline ini akan membaca dan menghasilkan data dalam struktur folder berikut. Pastikan Anda memulai dengan folder `annotated_dataset/` yang sudah terisi.
```
/proyek-rugae/ 
| 
|-- 📄 preprocess_final.py 
|-- 📄 augment_dataset.py 
|-- 📄 train_triplet.py 
|-- 📄 evaluate_final_simple.py 
|-- 📄 predict_triplet.py 
|-- 📄 dataset.py # (Modul pendukung) 
|-- 📄 model_recognition.py # (Modul pendukung) 
| 
|-- 📂 annotated_dataset/ # INPUT AWAL 
| 
|-- 📂 final_dataset_for_training/ # OUTPUT LANGKAH 1 
| 
|-- 📂 augmented_dataset_for_training/ # OUTPUT LANGKAH 2 
| 
|-- 📂 checkpoints/ # OUTPUT LANGKAH 3 
| 
|-- 📂 evaluation_results/ # OUTPUT LANGKAH 4
```

---
## 🚀 Langkah-langkah Eksekusi

Jalankan skrip-skrip berikut secara berurutan dari terminal Anda.

### **Langkah 1: Finalisasi & Normalisasi Dataset**

Langkah ini mengubah data anotasi mentah (yang masih miring) menjadi dataset `.npz` yang bersih, ternormalisasi posisinya, dan terstruktur dengan subfolder pasien.

* **Tujuan:** Menghasilkan dataset dasar dengan 1 sampel per pasien, sudah dalam posisi lurus dan siap untuk di-augmentasi.
* **Perintah:**
    ```bash
    python preprocess_final.py
    ```
* **Output:** Folder `final_dataset_for_training/` akan dibuat dan diisi dengan subfolder `train/`, `val/`, dan `test/`. Di dalamnya, setiap pasien akan memiliki subfolder sendiri yang berisi 1 file `.npz`.

### **Langkah 2: Augmentasi Dataset (Memperbanyak Sampel)**

Langkah ini sangat penting untuk memenuhi syarat `TripletLoss` yang membutuhkan lebih dari satu sampel per pasien.

* **Tujuan:** Membuat variasi artifisial (rotasi kecil & *jitter*) dari setiap sampel untuk memperkaya dataset.
* **Perintah:**
    ```bash
    python augment_dataset.py
    ```
* **Output:** Folder `augmented_dataset_for_training/` akan dibuat. Strukturnya sama seperti output sebelumnya, namun sekarang setiap subfolder pasien berisi **10 file `.npz`** (1 asli + 9 augmentasi).

### **Langkah 3: Pelatihan Model (Inti)**

Ini adalah proses utama di mana model AI akan dilatih menggunakan `TripletLoss`.

* **Tujuan:** Melatih model *recognition* pada dataset yang sudah di-augmentasi.
* **Perintah:**
    ```bash
    python train_triplet.py
    ```
* **Catatan:**
    * Pastikan variabel `data_root` di dalam skrip menunjuk ke `augmented_dataset_for_training`.
    * Proses ini akan memakan waktu paling lama. Anda bisa memantau `Train Loss` dan `Val Loss` di terminal.
* **Output:** File model terbaik akan disimpan sebagai `best_recognition_triplet_model.pth` di dalam folder `checkpoints/`.

### **Langkah 4: Evaluasi & Uji Coba**

Setelah model terlatih, kita bisa mengukur performanya pada `test set`.

* **Tujuan:** Melakukan analisis kuantitatif dan kualitatif pada performa model.
* **Perintah (Evaluasi Kuantitatif Lengkap):**
    ```bash
    python evaluate_final_simple.py
    ```
    * **Output:** Skrip ini akan menghasilkan **angka Akurasi Verifikasi**, **EER**, dan gambar **Confusion Matrix** (`confusion_matrix.png`) serta **Kurva EER** (`eer_curve.png`) di dalam folder `evaluation_results/`.

* **Perintah (Uji Coba Kualitatif Cepat):**
    ```bash
    python predict_triplet.py
    ```
    * **Output:** Skrip ini akan menjalankan perbandingan acak antara satu pasangan positif dan satu pasangan negatif, lalu mencetak hasilnya di terminal. Berguna untuk *spot-checking*.

---
## ℹ️ Keterangan Skrip Pendukung

Dua file berikut **tidak perlu dijalankan secara langsung**, namun sangat penting karena diimpor oleh skrip-skrip utama:

* **`dataset.py`**: Bertugas sebagai "pelayan" yang memuat data `.npz` dari disk dan menyajikannya ke PyTorch dalam format yang benar.
* **`model_recognition.py`**: Berisi "blueprint" atau arsitektur dari "otak" AI (PointNet++) yang kita latih.

---
## ⚠️ Troubleshooting

* **`CUDA out of memory`:** Jika Anda mendapatkan error ini saat training (`train_triplet.py`), buka skrip tersebut dan kurangi nilai `P` atau `K` di bagian `CONFIG` (misalnya, `P=2, K=2` untuk `batch_size=4`). Jika masih error, kurangi `TARGET_POINTS` di `finalize_data.py` (misal ke `20000`) lalu jalankan ulang Langkah 1 dan 2.
* **`FileNotFoundError`:** Pastikan Anda menjalankan skrip dari direktori yang benar dan struktur folder input sudah sesuai.
* **`Loss` selalu 0:** Pastikan Anda menjalankan Langkah 2 (augmentasi) dan `data_root` di `train_triplet.py` sudah menunjuk ke `augmented_dataset_for_training`.
