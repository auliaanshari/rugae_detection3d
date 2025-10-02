import open3d as o3d
import numpy as np
import os
import glob
import random

# --- KONFIGURASI ---
INPUT_DIR_ANNOTATED = "annotated_dataset"
OUTPUT_DIR_FINAL = "final_dataset_for_training"
TARGET_POINTS = 40000  # Jumlah titik final yang diinginkan

# Konfigurasi Pemisahan Dataset
TRAIN_SPLIT = 0.8  # 80% untuk data training
VALIDATION_SPLIT = 0.1 # 10% untuk data validasi
# Sisanya (10%) akan menjadi data test
# --------------------

def normalize_with_landmarks(mesh, landmarks):
    """
    Melakukan normalisasi pose (translasi & rotasi) pada mesh
    berdasarkan 3 titik landmark anatomis.
    """
    # Titik A (Papilla Insisivus) akan menjadi origin baru
    p_a = landmarks[0]
    # Titik B (Posterior Raphe)
    p_b = landmarks[1]
    # Titik C (Midpoint Raphe)
    p_c = landmarks[2]

    # --- 1. Translasi ---
    # Pindahkan mesh sehingga Papilla Insisivus (Titik A) berada di (0,0,0)
    mesh.translate(-p_a, relative=False)

    # --- 2. Rotasi ---
    # Definisikan sistem koordinat anatomis yang baru
    # Sumbu Y baru adalah vektor dari A ke B (garis tengah palatum)
    new_y_axis = (p_b - p_a) / np.linalg.norm(p_b - p_a)
    
    # Sumbu Z baru tegak lurus terhadap bidang yang dibentuk oleh A, B, dan C
    vec_ac = p_c - p_a
    new_z_axis = np.cross(new_y_axis, vec_ac)
    new_z_axis /= np.linalg.norm(new_z_axis)
    
    # Sumbu X baru tegak lurus terhadap Y dan Z
    new_x_axis = np.cross(new_y_axis, new_z_axis)

    # Buat matriks rotasi untuk menyelaraskan sumbu baru dengan sumbu dunia (X,Y,Z)
    # Ini adalah invers (transpose) dari matriks yang dibentuk oleh sumbu2 baru
    rotation_matrix = np.array([new_x_axis, new_y_axis, new_z_axis])
    
    # Terapkan rotasi
    mesh.rotate(rotation_matrix.T, center=(0,0,0))
    
    return mesh


def process_final_file(annotated_path, landmark_path, output_path_npz):
    """
    Memproses satu set data anotasi: normalisasi, downsampling, dan konversi ke .npz
    VERSI BARU: Membaca langsung sebagai Point Cloud untuk keandalan pembacaan label.
    """
    try:
        # 1. Memuat langsung sebagai Point Cloud, bukan Triangle Mesh
        pcd_annotated = o3d.io.read_point_cloud(annotated_path)
        
        # 2. Memuat titik landmark
        landmarks = np.loadtxt(landmark_path, delimiter=',')
        
        # 3. Validasi file yang dimuat
        if not pcd_annotated.has_points() or not pcd_annotated.has_colors():
            print(f"  -> KRITIS: File tidak memiliki titik atau informasi warna (label). Cek ulang file dari CloudCompare.")
            return

        print(f"  -> Poin anotasi padat: {len(pcd_annotated.points)}")
        
        # 4. Normalisasi Presisi menggunakan Landmark (fungsi ini bekerja untuk mesh & point cloud)
        pcd_normalized = normalize_with_landmarks(pcd_annotated, landmarks)
        
        # 5. Ekstrak vertices dan label (sekarang dari .points dan .colors)
        points = np.asarray(pcd_normalized.points)
        labels = np.asarray(pcd_normalized.colors)[:, 0] # Ambil channel Merah (Red)
        
        # Konversi float [0,1] ke integer label.
        labels_int = np.rint(labels * 255).astype(np.uint8)

        # Memetakan kembali nilai warna ke label asli kita
        labels_final_remapped = np.zeros_like(labels_int)
        labels_final_remapped[labels_int == 127] = 1
        labels_final_remapped[labels_int == 255] = 2
        
        # Cek cepat jika array label masih kosong (sebagai pengaman tambahan)
        if labels_int.size == 0:
            print("  -> FATAL ERROR: Gagal mengekstrak label dari warna. Array label kosong.")
            return

        # Langkah Downsampling
        if len(points) > TARGET_POINTS:
            chosen_indices = np.random.choice(len(points), TARGET_POINTS, replace=False)
            points_final = points[chosen_indices, :]
            labels_final = labels_final_remapped[chosen_indices]
        else:
            points_final = points
            labels_final = labels_final_remapped
            print(f"  -> Peringatan: Jumlah titik ({len(points)}) lebih sedikit dari target.")

        print(f"  -> Poin final: {len(points_final)}")

        # Langkah Simpan ke format .npz
        np.savez_compressed(
            output_path_npz,
            points=points_final,
            labels=labels_final
        )
        print(f"  -> Berhasil disimpan di {output_path_npz}")
        return True

    except Exception as e:
        print(f"  -> TERJADI ERROR: {e}")
        return False

def main():
    print("--- Memulai Proses Finalisasi Data untuk Training ---")
    if not os.path.isdir(INPUT_DIR_ANNOTATED):
        print(f"Error: Direktori input '{INPUT_DIR_ANNOTATED}' tidak ditemukan.")
        return

    all_patient_folders = [d for d in os.listdir(INPUT_DIR_ANNOTATED) if os.path.isdir(os.path.join(INPUT_DIR_ANNOTATED, d))]
    random.shuffle(all_patient_folders)

    # Logika untuk membagi dataset
    train_count = int(len(all_patient_folders) * TRAIN_SPLIT)
    val_count = int(len(all_patient_folders) * VALIDATION_SPLIT)
    
    train_patients = all_patient_folders[:train_count]
    val_patients = all_patient_folders[train_count : train_count + val_count]
    test_patients = all_patient_folders[train_count + val_count:]
    
    patient_splits = {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients
    }
    print(f"Dataset dibagi menjadi: {len(train_patients)} train, {len(val_patients)} validation, {len(test_patients)} test.")

    for subset, patients in patient_splits.items():
        output_subset_path = os.path.join(OUTPUT_DIR_FINAL, subset)
        os.makedirs(output_subset_path, exist_ok=True)
        
        print(f"\n--- Memproses subset: {subset.upper()} ---")
        
        for folder_name in patients:
            input_folder_path = os.path.join(INPUT_DIR_ANNOTATED, folder_name)
            print(f"Memproses pasien: {folder_name}")

            # Cari file mesh anotasi dan landmark
            annotated_file = glob.glob(os.path.join(input_folder_path, "*_annotated.ply"))
            landmark_file = glob.glob(os.path.join(input_folder_path, "*_landmarks.txt"))

            if not annotated_file or not landmark_file:
                print(f"  -> Peringatan: File anotasi atau landmark tidak ditemukan di {folder_name}.")
                continue
            
            output_file_name = f"{folder_name}_final.npz"
            output_file_path = os.path.join(output_subset_path, output_file_name)
            
            process_final_file(annotated_file[0], landmark_file[0], output_file_path)

    print("\n--- Proses Finalisasi Data Selesai ---")
    print(f"Data siap training tersedia di folder: {OUTPUT_DIR_FINAL}")


if __name__ == "__main__":
    main()