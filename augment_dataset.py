import numpy as np
import os
import glob
import shutil

# --- KONFIGURASI AUGMENTASI ---
INPUT_DIR = "final_dataset_for_training"
OUTPUT_DIR = "augmented_dataset_for_training"
# Berapa banyak variasi baru yang ingin dibuat dari setiap sampel asli
NUM_AUGMENTATIONS_PER_FILE = 9 # (Total akan ada 10 sampel per pasien: 1 asli + 9 augmentasi)

# Parameter untuk augmentasi
ROTATION_RANGE_DEG = [-10, 10] # Putar secara acak antara -10 hingga +10 derajat
JITTER_STRENGTH = 0.02 # Seberapa kuat "getaran" acak yang ditambahkan
# -----------------------------

def augment_points(points, rotation_range_deg=[-10, 10], jitter_strength=0.02):
    """
    Melakukan augmentasi rotasi acak (di sekitar sumbu Y) dan jitter pada point cloud.
    """
    # 1. Rotasi Acak di sekitar sumbu Y
    angle_deg = np.random.uniform(rotation_range_deg[0], rotation_range_deg[1])
    angle_rad = np.deg2rad(angle_deg)
    
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    
    # Matriks rotasi sumbu Y
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
    rotated_points = np.dot(points, R)

    # 2. Jitter Acak (menambah noise kecil)
    jitter = np.random.normal(0, jitter_strength, rotated_points.shape)
    jittered_points = rotated_points + jitter
    
    return jittered_points

def main():
    print("--- Memulai Proses Augmentasi Dataset Offline ---")
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Direktori input '{INPUT_DIR}' tidak ditemukan.")
        return

    # Buat ulang struktur folder train/val/test
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR) # Hapus folder lama jika ada
        print(f"Menghapus direktori lama: {OUTPUT_DIR}")

    for subset in ['train', 'val', 'test']:
        subset_path = os.path.join(INPUT_DIR, subset)
        if os.path.isdir(subset_path):
            os.makedirs(os.path.join(OUTPUT_DIR, subset), exist_ok=True)
    
    search_pattern = os.path.join(INPUT_DIR, "**", "*.npz")
    npz_files = glob.glob(search_pattern, recursive=True)
    
    for file_path in npz_files:
        relative_path = os.path.relpath(file_path, INPUT_DIR)
        output_dir_subset = os.path.dirname(os.path.join(OUTPUT_DIR, relative_path))
        base_name = os.path.basename(file_path)

        # Secara eksplisit membuat subfolder pasien di direktori output jika belum ada
        os.makedirs(output_dir_subset, exist_ok=True)
        
        print(f"\nMemproses: {relative_path}")
        
        # 1. Salin file aslinya terlebih dahulu
        shutil.copy(file_path, os.path.join(output_dir_subset, base_name))
        print(f"  -> Menyalin file asli.")
        
        # Muat data asli
        data = np.load(file_path)
        points_original = data['points']
        labels_original = data['labels']
        
        # 2. Buat file-file augmentasi baru
        for i in range(NUM_AUGMENTATIONS_PER_FILE):
            points_augmented = augment_points(points_original, ROTATION_RANGE_DEG, JITTER_STRENGTH)
            
            output_name = base_name.replace('.npz', f'_aug_{i+1}.npz')
            output_path = os.path.join(output_dir_subset, output_name)
            
            np.savez_compressed(
                output_path,
                points=points_augmented,
                labels=labels_original
            )
            print(f"  -> Membuat augmentasi #{i+1}")

    print(f"\n--- Proses Augmentasi Selesai. Dataset baru ada di folder: {OUTPUT_DIR} ---")


if __name__ == '__main__':
    main()