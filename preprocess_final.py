import open3d as o3d
import numpy as np
import os
import glob
import random

# --- KONFIGURASI ---
INPUT_DIR_ANNOTATED = "annotated_dataset"
OUTPUT_DIR_FINAL = "final_dataset_for_training"
TARGET_POINTS = 40000  # Jumlah titik final yang diinginkan ( ubah ini untuk menaikan sampling)

# Konfigurasi Pemisahan Dataset
TRAIN_SPLIT = 0.8  # 80% untuk data training
VALIDATION_SPLIT = 0.1 # 10% untuk data validasi
# Sisanya (10%) akan menjadi data test
# --------------------

def normalize_with_landmarks(mesh, landmarks):
    """
    VERSI FINAL YANG BEKERJA:
    Menggunakan 2 landmark (Depan & Belakang) dan referensi dunia untuk hasil lurus.
    """
    # Ambil hanya Landmark A (Depan) dan B (Belakang)
    p_a = landmarks[0]
    p_b = landmarks[1]
    
    # --- TAHAP 1: TRANSLASI ---
    mesh.translate(-p_a, relative=False)
    p_b_t = p_b - p_a

    # --- TAHAP 2: ROTASI ---
    # 2a. Tentukan Sumbu Y Anatomis (Arah Belakang)
    y_axis_anatomis = p_b_t / np.linalg.norm(p_b_t)
    
    # 2b. Tentukan Sumbu X Anatomis (Arah Kanan)
    world_z_ref = np.array([0, 0, 1])
    x_axis_anatomis = np.cross(y_axis_anatomis, world_z_ref)
    x_axis_anatomis /= np.linalg.norm(x_axis_anatomis)

    # 2c. Tentukan Sumbu Z Anatomis (Arah Atas)
    z_axis_anatomis = np.cross(x_axis_anatomis, y_axis_anatomis)
    
    # 2d. Susun Matriks Rotasi
    rotation_matrix = np.array([x_axis_anatomis, y_axis_anatomis, z_axis_anatomis])

    # 2e. Terapkan Rotasi
    mesh.rotate(rotation_matrix, center=(0,0,0))
    
    # --- TAHAP 3: FINALISASI VISUAL ---
    mesh.orient_triangles()
    mesh.compute_vertex_normals()
    
    return mesh


def process_final_file(annotated_path, landmark_path, output_path_npz):
    """
    Memproses satu set data anotasi: normalisasi, downsampling, dan konversi ke .npz
    VERSI BARU: Membaca langsung sebagai Point Cloud untuk keandalan pembacaan label.
    """
    try:
        # 1. Memuat langsung sebagai Triangle Mesh
        mesh_annotated = o3d.io.read_triangle_mesh(annotated_path)
        
        # 2. Memuat titik landmark
        landmarks = np.loadtxt(landmark_path, delimiter=',')
        
        # 3. Validasi file yang dimuat
        if not mesh_annotated.has_vertices() or not mesh_annotated.has_vertex_colors():
            print(f"  -> KRITIS: File tidak memiliki titik atau informasi warna (label). Cek ulang file dari CloudCompare.")
            return

        print(f"  -> Poin anotasi padat: {len(mesh_annotated.vertices)}")
        
        # 4. Normalisasi Presisi menggunakan Landmark (fungsi ini bekerja untuk mesh & point cloud)
        mesh_normalized = normalize_with_landmarks(mesh_annotated, landmarks)

        # 5. Ekstrak vertices dan label (sekarang dari .points dan .colors)
        points = np.asarray(mesh_normalized.vertices)
        labels = np.asarray(mesh_normalized.vertex_colors)[:, 0] # Ambil channel Merah (Red)

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
    train_patients, val_patients, test_patients = np.split(np.array(all_patient_folders), [train_count, train_count + val_count])
    
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

            # Buat subfolder untuk pasien di dalam folder train/val/test
            output_patient_folder = os.path.join(output_subset_path, folder_name)
            os.makedirs(output_patient_folder, exist_ok=True)

            # Cari file mesh anotasi dan landmark
            annotated_file = glob.glob(os.path.join(input_folder_path, "*_annotated.ply"))
            landmark_file = glob.glob(os.path.join(input_folder_path, "*_landmarks.txt"))

            if not annotated_file or not landmark_file:
                print(f"  -> Peringatan: File anotasi atau landmark tidak ditemukan di {folder_name}.")
                continue
            
            output_file_name = f"{folder_name}_final.npz"
            output_file_path = os.path.join(output_patient_folder, output_file_name)
            
            process_final_file(annotated_file[0], landmark_file[0], output_file_path)

    print("\n--- Proses Finalisasi Data Selesai ---")
    print(f"Data siap training tersedia di folder: {OUTPUT_DIR_FINAL}")


if __name__ == "__main__":
    main()