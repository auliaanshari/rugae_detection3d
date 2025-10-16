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
            
        self.file_paths = sorted(glob.glob(os.path.join(self.split_dir, "**", "*.npz"), recursive=True))
        
        if len(self.file_paths) == 0:
            print(f"Peringatan: Tidak ada file .npz yang ditemukan di {self.split_dir}")
        
        # Membuat pemetaan dari ID pasien (string) ke indeks kelas (integer)
        all_patient_ids = sorted(list(set([os.path.basename(os.path.dirname(p)) for p in self.file_paths])))
        self.classes = all_patient_ids
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}
        self.idx_to_class = {i: class_name for i, class_name in enumerate(self.classes)}

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
        segmentation_labels = data['labels']
        
        # Konversi array NumPy ke format PyTorch Tensor
        # .float() untuk koordinat, .long() untuk label kelas
        points = torch.from_numpy(points).float()
        segmentation_labels = torch.from_numpy(segmentation_labels).long()
        
        # Ekstrak ID pasien dari nama file untuk digunakan nanti (di model recognition)
        # Contoh nama file: 'K01A (Bernado Barus)_final.npz' -> 'K01A (Bernado Barus)'
        patient_id_str = os.path.basename(os.path.dirname(file_path))

        # Ambil indeks kelas integer untuk ID pasien ini
        class_idx = self.class_to_idx[patient_id_str]
        
        # Kembalikan data dalam bentuk dictionary
        sample = {
            'points': points, 
            'segmentation_labels': segmentation_labels, 
            'patient_id': patient_id_str,
            'class_idx': torch.tensor(class_idx, dtype=torch.long)
        }
        
        return sample

# Kode di bawah ini hanya akan berjalan jika Anda menjalankan 'python dataset.py'
# secara langsung. Ini tidak akan berjalan saat Anda mengimpor class ini di file lain.
if __name__ == '__main__':
    print("--- Menjalankan Tes untuk RugaeDataset ---")
    
    # Ganti 'train' dengan 'val' atau 'test' jika folder tersebut sudah ada isinya
    try:
        train_dataset = RugaeDataset(root_dir="final_dataset_for_training", split='train')
        
        print(f"Total sampel di dataset 'train': {len(train_dataset)}")

        # Tampilkan pemetaan kelas yang dibuat
        print(f"Jumlah kelas (pasien unik) ditemukan: {len(train_dataset.classes)}")
        print("Pemetaan Kelas ke Indeks:")
        print(train_dataset.class_to_idx)
        
        if len(train_dataset) > 0:
            # Ambil sampel pertama untuk diinspeksi
            first_sample = train_dataset[0]
            
            print("\n--- Inspeksi Sampel Pertama ---")
            print(f"ID Pasien (string): {first_sample['patient_id']}")
            print(f"Indeks Kelas (integer): {first_sample['class_idx'].item()}")
            print(f"Bentuk 'points': {first_sample['points'].shape}")
            print(f"Bentuk 'segmentation_labels': {first_sample['segmentation_labels'].shape}")
            print("-" * 20)
            
            # Cek apakah DataLoader bekerja
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
            first_batch = next(iter(train_loader))
            print("\n--- Inspeksi Batch Pertama dari DataLoader ---")
            print(f"Bentuk 'points' dalam satu batch: {first_batch['points'].shape}")
            print(f"Bentuk 'segmentation_labels' dalam satu batch: {first_batch['segmentation_labels'].shape}")
            print(f"ID Pasien dalam satu batch: {first_batch['patient_id']}")
            print("-" * 20)

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Pastikan Anda sudah menjalankan 'preprocess_final.py' dan folder 'final_dataset_for_training/train' sudah ada isinya.")