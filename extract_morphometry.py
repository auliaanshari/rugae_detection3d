import numpy as np
import os
import glob
from skimage.morphology import skeletonize
from scipy.ndimage import label, binary_dilation
import pandas as pd

# --- KONFIGURASI ---
INPUT_DIR_FINAL = "final_dataset_for_training"
OUTPUT_DIR_MORPH = "morphometry_results"
# Ukuran voxel dalam mm. HARUS SESUAI dengan skala data Anda.
# Jika Anda tidak melakukan normalisasi skala, ini adalah ukuran dalam mm.
VOXEL_SIZE = 0.3 
# --------------------


def voxelize_point_cloud(points, voxel_size):
    """Mengubah point cloud menjadi 3D voxel grid yang biner."""
    # Cari batas minimum dan maksimum dari point cloud
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    # Tentukan dimensi grid
    dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int) + 1
    
    # Buat grid kosong
    voxel_grid = np.zeros(dims, dtype=bool)
    
    # Tempatkan setiap titik ke dalam voxel yang sesuai
    indices = np.floor((points - min_bound) / voxel_size).astype(int)
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    
    return voxel_grid


def extract_features_from_npz(npz_path, output_csv_path):
    """
    Fungsi utama untuk mengekstrak fitur morfometrik dari satu file .npz.
    """
    try:
        data = np.load(npz_path)
        points = data['points']
        labels = data['labels']
        
        # 1. Isolasi titik-titik yang merupakan Rugae (label=2)
        rugae_points = points[labels == 255]
        
        if rugae_points.shape[0] < 10:
            print(f"  -> Tidak ada atau terlalu sedikit titik rugae ditemukan. Dilewati.")
            return

        # 2. Voxelization: Ubah point cloud rugae menjadi struktur balok 3D
        rugae_voxel_grid = voxelize_point_cloud(rugae_points, VOXEL_SIZE)
        
        # Menebalkan grid satu lapis untuk menyambungkan fragmen-fragmen yang berdekatan
        dilated_grid = binary_dilation(rugae_voxel_grid, iterations=1)

        # 3. Pisahkan setiap gundukan rugae yang tidak terhubung
        # labeled_array akan berisi ID unik untuk setiap gundukan (1, 2, 3, ...)
        # num_features adalah jumlah total rugae yang ditemukan
        labeled_array, num_features = label(dilated_grid)
        
        print(f"  -> Ditemukan {num_features} gundukan rugae.")
        
        results = []
        for i in range(1, num_features + 1):
            # Ambil satu gundukan rugae saja
            single_rugae_grid = (labeled_array == i)
            
            # 4. Skeletonization: Cari 'tulang punggung' dari gundukan rugae
            skeleton = skeletonize(single_rugae_grid)
            
            # 5. Hitung Panjang (Estimasi)
            # Panjang diestimasi dari jumlah voxel di skeleton dikali ukuran voxel
            num_skeleton_voxels = np.sum(skeleton)
            length_mm = num_skeleton_voxels * VOXEL_SIZE
            
            results.append({
                "rugae_id": i,
                "length_mm": round(length_mm, 2),
                "skeleton_voxels": num_skeleton_voxels
            })

        # 6. Simpan hasil ke file CSV
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_csv_path, index=False)
            print(f"  -> Hasil morfometrik disimpan di {output_csv_path}")
        
    except Exception as e:
        print(f"  -> TERJADI ERROR: {e}")


def main():
    print("--- Memulai Proses Ekstraksi Morfometrik ---")
    if not os.path.isdir(INPUT_DIR_FINAL):
        print(f"Error: Direktori input '{INPUT_DIR_FINAL}' tidak ditemukan.")
        return

    os.makedirs(OUTPUT_DIR_MORPH, exist_ok=True)
    
    # Cari semua file .npz di dalam folder train, val, dan test
    search_pattern = os.path.join(INPUT_DIR_FINAL, "**", "*.npz")
    npz_files = glob.glob(search_pattern, recursive=True)
    
    for npz_file_path in npz_files:
        # Dapatkan nama file dasar untuk output
        base_name = os.path.basename(npz_file_path).replace('.npz', '')
        # Dapatkan nama pasien dari path
        patient_name = os.path.basename(os.path.dirname(npz_file_path))
        
        print(f"\nMemproses: {patient_name}/{base_name}")
        
        output_csv_name = f"{base_name}_morphometry.csv"
        # Buat subfolder di output agar rapi
        output_patient_folder = os.path.join(OUTPUT_DIR_MORPH, patient_name)
        os.makedirs(output_patient_folder, exist_ok=True)
        output_csv_path = os.path.join(output_patient_folder, output_csv_name)
        
        extract_features_from_npz(npz_file_path, output_csv_path)

    print(f"\n--- Proses Ekstraksi Selesai. Hasil ada di folder: {OUTPUT_DIR_MORPH} ---")

if __name__ == "__main__":
    main()