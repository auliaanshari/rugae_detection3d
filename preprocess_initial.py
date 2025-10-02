import open3d as o3d
import numpy as np
import os
import glob

# --- KONFIGURASI ---
INPUT_DIR = "dataset"
OUTPUT_DIR = "to_be_annotated_mesh"
# --------------------

def process_initial_mesh(input_path, output_path):
    """
    Memuat file mesh, membersihkan, menormalisasi pose, dan menyimpan
    kembali sebagai mesh yang solid.
    """
    try:
        # Memuat file sebagai TriangleMesh
        mesh = o3d.io.read_triangle_mesh(input_path)
        if not mesh.has_vertices():
            print(f"  -> Gagal memuat atau mesh kosong.")
            return

        print(f"  -> Vertex asli: {len(mesh.vertices)}")
        
        # Langkah 1: Pembersihan Mesh (bukan outlier removal point cloud)
        # Menghapus segitiga yang "rusak" atau tidak valid
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        print(f"  -> Vertex setelah pembersihan: {len(mesh.vertices)}")

        # Langkah 2: Normalisasi Pose pada Mesh
        # a. Translasi ke pusat (0,0,0)
        center = mesh.get_center()
        mesh.translate(-center, relative=False)

        # b. Orientasi menggunakan PCA
        # Kita terapkan transformasi pada vertex-vertex dari mesh
        vertices = np.asarray(mesh.vertices)
        covariance_matrix = np.cov(vertices, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        rotation_matrix = eigenvectors.T
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, -1] *= -1
        
        # Terapkan rotasi pada mesh
        mesh.rotate(rotation_matrix, center=(0,0,0))

        # Langkah 3: Simpan sebagai file mesh
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"  -> Berhasil disimpan sebagai MESH di {output_path}")
        return True

    except Exception as e:
        print(f"  -> TERJADI ERROR: {e}")
        return False

def main():
    print("--- Memulai Proses Preprocessing Awal (Mode Mesh) ---")
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Direktori input '{INPUT_DIR}' tidak ditemukan.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    patient_folders = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    for folder_name in patient_folders:
        input_folder_path = os.path.join(INPUT_DIR, folder_name)
        output_folder_path = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)
        
        print(f"\nMemproses folder: {folder_name}")
        
        file_to_process = None
        for ext in ["-UpperJaw.ply"]:
            search_pattern = os.path.join(input_folder_path, f"*{ext}")
            found_files = glob.glob(search_pattern)
            if found_files:
                file_to_process = found_files[0]
                break
        
        if not file_to_process:
            print(f"  -> Peringatan: Tidak ditemukan file UpperJaw di dalam {folder_name}.")
            continue
            
        output_file_name = f"{folder_name}_cleaned.ply"
        output_file_path = os.path.join(output_folder_path, output_file_name)
        
        process_initial_mesh(file_to_process, output_file_path)

    print("\n--- Proses Preprocessing Awal (Mode Mesh) Selesai ---")

if __name__ == "__main__":
    main()