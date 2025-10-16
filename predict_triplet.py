import torch
import numpy as np
import os
import glob
import random
from model_recognition import PointNet2Recognition

# --- KONFIGURASI ---
CONFIG = {
    "embedding_dim": 256,
    "model_path": "checkpoints/best_recognition_triplet_model.pth",
    "data_root": "augmented_dataset_for_training/test" # Menguji pada data test
}
# --------------------

def load_sample_for_prediction(file_path, device):
    """
    Memuat satu file .npz dan menyiapkannya untuk input model.
    """
    data = np.load(file_path)
    points = data['points']
    
    # Konversi ke PyTorch Tensor
    points_tensor = torch.from_numpy(points).float()
    
    # Ubah format dari (N, 3) menjadi (1, 3, N) sesuai input model
    points_tensor = points_tensor.unsqueeze(0)
    
    return points_tensor.to(device)

def main():
    print("--- Memulai Uji Coba Model Triplet ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # 1. Muat arsitektur model
    model = PointNet2Recognition(embedding_dim=CONFIG["embedding_dim"]).to(device)
    
    # 2. Muat bobot (weights) yang sudah terlatih
    if not os.path.exists(CONFIG["model_path"]):
        print(f"Error: File model tidak ditemukan di '{CONFIG['model_path']}'")
        return
        
    model.load_state_dict(torch.load(CONFIG["model_path"]))
    model.eval() # Set model ke mode evaluasi (penting!)
    print(f"Model berhasil dimuat dari: {CONFIG['model_path']}")

    # 3. Cari sampel untuk diuji dari folder test
    all_patient_folders = [d for d in os.listdir(CONFIG["data_root"]) if os.path.isdir(os.path.join(CONFIG["data_root"], d))]
    
    if len(all_patient_folders) < 2:
        print("Error: Butuh minimal 2 pasien di folder test untuk melakukan perbandingan.")
        return

    # --- UJI COBA 1: PASANGAN POSITIF (Pasien yang sama) ---
    print("\n--- Menguji Pasangan Positif (pasien sama) ---")
    
    # Pilih satu pasien secara acak
    patient_a_folder_name = random.choice(all_patient_folders)
    patient_a_path = os.path.join(CONFIG["data_root"], patient_a_folder_name)
    patient_a_samples = glob.glob(os.path.join(patient_a_path, "*.npz"))

    if len(patient_a_samples) < 2:
        print(f"Peringatan: Tidak bisa menemukan 2 sampel untuk pasien {patient_a_folder_name}, mencari lagi...")
        # (Logika sederhana untuk mencari lagi, bisa diperbaiki jika perlu)
        return

    # Ambil dua sampel berbeda dari pasien yang sama
    sample_a1_path, sample_a2_path = random.sample(patient_a_samples, 2)
    
    print(f"Membandingkan:")
    print(f"  - Sampel 1: {os.path.basename(sample_a1_path)}")
    print(f"  - Sampel 2: {os.path.basename(sample_a2_path)}")

    # Muat sampel
    sample_a1_tensor = load_sample_for_prediction(sample_a1_path, device)
    sample_a2_tensor = load_sample_for_prediction(sample_a2_path, device)

    # Dapatkan embedding
    with torch.no_grad():
        emb_a1 = model(sample_a1_tensor)
        emb_a2 = model(sample_a2_tensor)

    # Hitung jarak Euclidean
    dist_positive = torch.nn.functional.pairwise_distance(emb_a1, emb_a2).item()
    print(f"\n>>> Jarak Embedding (Positif): {dist_positive:.4f}")

    # --- UJI COBA 2: PASANGAN NEGATIF (Pasien berbeda) ---
    print("\n--- Menguji Pasangan Negatif (pasien berbeda) ---")
    
    # Pilih pasien lain yang berbeda
    patient_b_folder_name = random.choice([p for p in all_patient_folders if p != patient_a_folder_name])
    patient_b_path = os.path.join(CONFIG["data_root"], patient_b_folder_name)
    sample_b1_path = random.choice(glob.glob(os.path.join(patient_b_path, "*.npz")))

    print(f"Membandingkan:")
    print(f"  - Sampel 1: {os.path.basename(sample_a1_path)} (dari Pasien A)")
    print(f"  - Sampel 2: {os.path.basename(sample_b1_path)} (dari Pasien B)")
    
    # Muat sampel
    sample_b1_tensor = load_sample_for_prediction(sample_b1_path, device)

    # Dapatkan embedding
    with torch.no_grad():
        emb_b1 = model(sample_b1_tensor)

    # Hitung jarak Euclidean
    dist_negative = torch.nn.functional.pairwise_distance(emb_a1, emb_b1).item()
    print(f"\n>>> Jarak Embedding (Negatif): {dist_negative:.4f}")

    # --- KESIMPULAN ---
    print("\n--- Kesimpulan ---")
    print(f"Jarak Positif (pasien sama): {dist_positive:.4f}")
    print(f"Jarak Negatif (pasien beda): {dist_negative:.4f}")
    
    if dist_positive < dist_negative:
        print("✅ Berhasil: Jarak Positif < Jarak Negatif. Model dapat membedakan identitas.")
    else:
        print("❌ Gagal: Jarak Positif >= Jarak Negatif. Model masih bingung.")

if __name__ == "__main__":
    main()