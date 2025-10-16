import torch
import numpy as np
import os
import glob
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

from model_recognition import PointNet2Recognition

# --- KONFIGURASI ---
CONFIG = {
    "embedding_dim": 256,
    "model_path": "checkpoints/best_recognition_triplet_model.pth",
    "data_root": "augmented_dataset_for_training/test",
    "output_dir": "evaluation_results",
    "SAMPLES_PER_PATIENT": 5 # Ambil 5 sampel acak per pasien untuk evaluasi
}
# --------------------

def load_sample_for_prediction(file_path, device):
    """ Memuat satu file .npz dan menyiapkannya untuk input model. """
    data = np.load(file_path)
    points = data['points']
    points_tensor = torch.from_numpy(points).float().unsqueeze(0)
    return points_tensor.to(device)

def main():
    print("--- Memulai Evaluasi Final (Kuantitatif & Seimbang) ---")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Muat model terlatih
    model = PointNet2Recognition(embedding_dim=CONFIG["embedding_dim"]).to(device)
    model.load_state_dict(torch.load(CONFIG["model_path"]))
    model.eval()
    print("Model berhasil dimuat.")

    # 2. Hasilkan embedding untuk sampel yang DIPILIH dari test set
    print(f"\nMengambil {CONFIG['SAMPLES_PER_PATIENT']} sampel acak per pasien...")
    all_embeddings = defaultdict(list)
    
    patient_folders = sorted([d for d in os.listdir(CONFIG["data_root"]) if os.path.isdir(os.path.join(CONFIG["data_root"], d))])

    with torch.no_grad():
        for patient_id in tqdm(patient_folders, desc="Generating Embeddings"):
            patient_path = os.path.join(CONFIG["data_root"], patient_id)
            all_files = glob.glob(os.path.join(patient_path, "*.npz"))
            
            num_to_sample = min(CONFIG['SAMPLES_PER_PATIENT'], len(all_files))
            selected_files = random.sample(all_files, num_to_sample)

            for file_path in selected_files:
                points_tensor = load_sample_for_prediction(file_path, device)
                embedding = model(points_tensor).cpu().numpy()
                all_embeddings[patient_id].append(embedding)

    # 3. Buat Pasangan Positif dan Negatif yang Seimbang
    print("\nMembuat pasangan uji yang seimbang...")
    
    # Pasangan Positif
    positive_pairs = []
    for patient_id in patient_folders:
        embeddings = all_embeddings[patient_id]
        if len(embeddings) >= 2:
            for i, j in combinations(range(len(embeddings)), 2):
                positive_pairs.append((embeddings[i], embeddings[j]))

    # Pasangan Negatif
    negative_pairs = []
    all_embeddings_list = []
    all_labels_list = []
    for i, patient_id in enumerate(patient_folders):
        for emb in all_embeddings[patient_id]:
            all_embeddings_list.append(emb)
            all_labels_list.append(i)

    # Buat pasangan negatif hingga jumlahnya sama dengan pasangan positif
    target_num_negatives = len(positive_pairs)
    while len(negative_pairs) < target_num_negatives:
        idx1, idx2 = random.sample(range(len(all_embeddings_list)), 2)
        if all_labels_list[idx1] != all_labels_list[idx2]:
            negative_pairs.append((all_embeddings_list[idx1], all_embeddings_list[idx2]))

    print(f"Dibuat {len(positive_pairs)} pasangan positif.")
    print(f"Dibuat {len(negative_pairs)} pasangan negatif.")

    # 4. Hitung semua jarak
    dist_pos = [np.linalg.norm(p[0] - p[1]) for p in positive_pairs]
    dist_neg = [np.linalg.norm(p[0] - p[1]) for p in negative_pairs]
    distances = np.concatenate([dist_pos, dist_neg])
    labels = np.concatenate([np.ones_like(dist_pos), np.zeros_like(dist_neg)]) # 1=Sama, 0=Beda

    # 5. Cari Threshold Terbaik
    print("\nMencari threshold terbaik untuk akurasi...")
    best_accuracy = 0
    best_threshold = 0
    
    thresholds = np.arange(min(distances), max(distances), 0.01)
    for threshold in thresholds:
        predictions = (distances < threshold)
        accuracy = np.mean(predictions == labels) * 100
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            
    # --- HASIL AKHIR ---
    print("\n--- Hasil Evaluasi Final ---")
    print(f"Jumlah Pasangan Positif\t\t: {len(positive_pairs)}")
    print(f"Jumlah Pasangan Negatif\t\t: {len(negative_pairs)}")
    print(f"Threshold Jarak Terbaik\t\t: {best_threshold:.4f}")
    print(f"Akurasi Verifikasi Final (Test Set)\t: {best_accuracy:.2f}%")

    # --- BLOK VISUALISASI CONFUSION MATRIX ---
    print("\nMembuat visualisasi confusion matrix...")
    
    # Buat prediksi final menggunakan threshold terbaik
    final_predictions = (distances < best_threshold)
    
    # Hitung confusion matrix
    cm = confusion_matrix(labels, final_predictions)
    
    # Buat plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Prediksi Beda', 'Prediksi Sama'], 
                yticklabels=['Aktual Beda', 'Aktual Sama'])
    plt.xlabel('Label Prediksi')
    plt.ylabel('Label Aktual')
    plt.title('Confusion Matrix untuk Verifikasi Identitas')
    
    output_path = os.path.join(CONFIG["output_dir"], "confusion_matrix.png")
    plt.savefig(output_path)
    print(f"✅ Confusion matrix berhasil disimpan di: {output_path}")
    # ----------------------------------------------------

    # --- BLOK MENGHITUNG EER DAN MEMBUAT PLOT ---
    print("\nMenghitung EER (Equal Error Rate)...")
    
    # PENTING: roc_curve butuh "skor", di mana nilai tinggi lebih baik.
    # Karena kita pakai jarak (nilai rendah lebih baik), kita gunakan negatif dari jarak.
    scores = -distances
    
    # Gunakan roc_curve dari scikit-learn
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr # False Negative Rate = 1 - True Positive Rate (juga dikenal sebagai FRR)
    
    # Cari titik di mana selisih antara fpr dan fnr paling kecil
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    
    # EER adalah nilai rata-rata dari fpr dan fnr di titik tersebut
    eer_value = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    
    print(f"Equal Error Rate (EER)\t\t: {eer_value * 100:.2f}%")
    print(f"Threshold Jarak untuk EER\t: {-eer_threshold:.4f}")
    
    # Buat plot kurva FAR vs FRR
    plt.figure()
    plt.plot(-thresholds, fpr * 100, label='FAR (False Acceptance Rate)') # Gunakan -thresholds agar kembali ke skala jarak
    plt.plot(-thresholds, fnr * 100, label='FRR (False Rejection Rate)')
    plt.xlabel('Threshold Jarak')
    plt.ylabel('Error Rate (%)')
    plt.title('Kurva FAR vs FRR untuk Menentukan EER')
    plt.legend()
    plt.grid(True)
    
    # Tandai titik EER
    plt.plot(-eer_threshold, eer_value * 100, 'ro', label=f'EER = {eer_value * 100:.2f}%')
    
    output_path_eer = os.path.join(CONFIG["output_dir"], "eer_curve.png")
    plt.savefig(output_path_eer)
    print(f"✅ Plot kurva EER berhasil disimpan di: {output_path_eer}")
    # ----------------------------------------------------

if __name__ == "__main__":
    main()