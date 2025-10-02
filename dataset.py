import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob

class RugaeDataset(Dataset):
    """
    Dataset kustom untuk memuat data point cloud rugae dari file .npz.
    """
    def __init__(self, root_dir="final_dataset_for_training", split='train'):
        """
        Inisialisasi dataset.
        Args:
            root_dir (string): Direktori yang berisi folder 'train', 'val', 'test'.
            split (string): Memilih antara 'train', 'val', atau 'test'.
        """
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(self.root_dir, self.split)
        
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Direktori untuk split '{self.split}' tidak ditemukan di: {self.split_dir}")
            
        self.file_paths = sorted(glob.glob(os.path.join(self.split_dir, "*.npz")))
        
        if len(self.file_paths) == 0:
            print(f"Peringatan: Tidak ada file .npz yang ditemukan di {self.split_dir}")

    def __len__(self):
        """Mengembalikan jumlah total sampel di dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Mengambil satu sampel data (point cloud dan labelnya) berdasarkan indeks.
        """
        # Muat data dari file .npz
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        
        points = data['points']
        labels = data['labels']
        
        # Konversi array NumPy ke format PyTorch Tensor
        # .float() untuk koordinat, .long() untuk label kelas
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()
        
        # Ekstrak ID pasien dari nama file untuk digunakan nanti (di model recognition)
        # Contoh nama file: 'K01A (Bernado Barus)_final.npz' -> 'K01A (Bernado Barus)'
        patient_id = os.path.basename(file_path).replace('_final.npz', '')
        
        # Kembalikan data dalam bentuk dictionary
        sample = {
            'points': points, 
            'labels': labels, 
            'patient_id': patient_id
        }
        
        return sample

# --- BLOK UNTUK PENGUJIAN ---
# Kode di bawah ini hanya akan berjalan jika Anda menjalankan 'python dataset.py'
# secara langsung. Ini tidak akan berjalan saat Anda mengimpor class ini di file lain.
if __name__ == '__main__':
    print("--- Menjalankan Tes untuk RugaeDataset ---")
    
    # Ganti 'train' dengan 'val' atau 'test' jika folder tersebut sudah ada isinya
    try:
        train_dataset = RugaeDataset(root_dir="final_dataset_for_training", split='train')
        
        print(f"Total sampel di dataset 'train': {len(train_dataset)}")
        
        if len(train_dataset) > 0:
            # Ambil sampel pertama untuk diinspeksi
            first_sample = train_dataset[0]
            
            points_tensor = first_sample['points']
            labels_tensor = first_sample['labels']
            patient_id = first_sample['patient_id']
            
            print("\n--- Inspeksi Sampel Pertama ---")
            print(f"ID Pasien: {patient_id}")
            print(f"Bentuk (Shape) tensor 'points': {points_tensor.shape}")
            print(f"Tipe data 'points': {points_tensor.dtype}")
            print(f"Bentuk (Shape) tensor 'labels': {labels_tensor.shape}")
            print(f"Tipe data 'labels': {labels_tensor.dtype}")
            print("-" * 20)
            
            # Cek apakah DataLoader bekerja
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
            first_batch = next(iter(train_loader))
            print("\n--- Inspeksi Batch Pertama dari DataLoader ---")
            print(f"Bentuk 'points' dalam satu batch: {first_batch['points'].shape}")
            print(f"Bentuk 'labels' dalam satu batch: {first_batch['labels'].shape}")
            print(f"ID Pasien dalam satu batch: {first_batch['patient_id']}")
            print("-" * 20)

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Pastikan Anda sudah menjalankan 'preprocess_final.py' dan folder 'final_dataset_for_training/train' sudah ada isinya.")