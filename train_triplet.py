import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import random

# Impor class yang sudah kita buat
from dataset import RugaeDataset
from model_recognition import PointNet2Recognition 

# --- KONFIGURASI PELATIHAN ---
CONFIG = {
    "learning_rate": 0.0001,
    # "batch_size": 4, # Triplet loss bekerja lebih baik dengan batch size lebih besar
    "num_epochs": 20,
    "embedding_dim": 256,
    "margin": 0.5, # Jarak minimal antara pasangan positif dan negatif
    # "data_root": "final_dataset_for_training",
    "data_root": "augmented_dataset_for_training",
    "output_dir": "checkpoints",
    # --- PENGATURAN SAMPLER BARU ---
    "P": 2, # Jumlah pasien unik per batch
    "K": 2  # Jumlah sampel per pasien ( ini bisa dinaikkan untuk batch size lebih besar)
}
# -----------------------------

# (Fungsi augmentasi `random_rotation_augmentation` tetap sama seperti sebelumnya)
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, P, K):
        self.dataset = dataset
        self.P = P
        self.K = K
        
        self.labels = np.array([d['class_idx'].item() for d in dataset])
        self.labels_to_indices = defaultdict(list)
        for i, label in enumerate(self.labels):
            self.labels_to_indices[label].append(i)
            
        self.unique_labels = list(self.labels_to_indices.keys())
        
        # Hitung jumlah batch per epoch
        self.num_batches = len(self.unique_labels) // self.P

    def __iter__(self):
        random.shuffle(self.unique_labels)
        for i in range(self.num_batches):
            batch_indices = []
            # Pilih P kelas/pasien secara acak
            selected_labels = self.unique_labels[i*self.P : (i+1)*self.P]
            
            for label in selected_labels:
                # Ambil K sampel secara acak dari setiap pasien
                indices = self.labels_to_indices[label]
                random.shuffle(indices)
                batch_indices.extend(indices[:self.K])
            
            yield batch_indices

    def __len__(self):
        return self.num_batches

def train_triplet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Persiapan Dataset
    train_dataset = RugaeDataset(root_dir=CONFIG["data_root"], split='train')
    val_dataset = RugaeDataset(root_dir=CONFIG["data_root"], split='val')

    train_batch_sampler = BalancedBatchSampler(train_dataset, P=CONFIG["P"], K=CONFIG["K"])
    val_batch_sampler = BalancedBatchSampler(val_dataset, P=CONFIG["P"], K=CONFIG["K"])

    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, num_workers=0)

    # Inisialisasi Model, Loss, dan Optimizer
    model = PointNet2Recognition(embedding_dim=CONFIG["embedding_dim"]).to(device)
    criterion = nn.TripletMarginLoss(margin=CONFIG["margin"], p=2.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    best_val_loss = float('inf')
    print(f"\n--- Memulai Pelatihan (Triplet Loss, P*K Sampler) ---")
    print(f"Batch size efektif: {CONFIG['P']} pasien x {CONFIG['K']} sampel = {CONFIG['P'] * CONFIG['K']} total")

    for epoch in range(CONFIG["num_epochs"]):
        # --- Fase Training ---
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Train]")
        for batch in progress_bar:
            points, class_labels = batch['points'].to(device), batch['class_idx'].to(device)
            
            optimizer.zero_grad()
            
            # Dapatkan embedding untuk seluruh batch
            embeddings = model(points)
            
            # --- Logika Online Triplet Mining (All-to-All) ---
            # 1. Hitung matriks jarak pairwise
            dist_matrix = torch.cdist(embeddings, embeddings, p=2.0)

            # 2. Cari pasangan positif dan negatif
            # Dapatkan matriks boolean: True jika label sama
            labels_equal = class_labels.view(-1, 1) == class_labels.view(1, -1)
            
            # Mask untuk pasangan positif (label sama, tapi bukan titik yang sama)
            positive_mask = labels_equal & ~torch.eye(labels_equal.shape[0], dtype=torch.bool, device=device)
            # Mask untuk pasangan negatif (label berbeda)
            negative_mask = ~labels_equal
            
            # 3. Pilih triplet dan hitung loss
            # Ambil semua jarak positif dan negatif yang valid
            positive_dists = dist_matrix[positive_mask]
            negative_dists = dist_matrix[negative_mask]
            
            # Untuk setiap anchor, kita butuh 1 positive dan 1 negative
            # Kita akan ulangi setiap jarak negatif untuk setiap jarak positif yang mungkin
            # Ini adalah pendekatan 'all-to-all' yang sederhana
            if len(positive_dists) > 0 and len(negative_dists) > 0:
                # anchor_positive_dists [P, 1] - anchor_negative_dists [1, N]
                loss_matrix = positive_dists.unsqueeze(1) - negative_dists.unsqueeze(0) + CONFIG["margin"]
                # Ambil hanya nilai loss yang > 0
                loss = F.relu(loss_matrix).mean()
            else:
                loss = torch.tensor(0.0, device=device) # Tidak ada triplet valid di batch ini
            
            # ----------------------------------------------------

            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Fase Validasi (hanya memantau loss) ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                points, class_labels = batch['points'].to(device), batch['class_idx'].to(device)
                embeddings = model(points)
                # Ulangi logika loss yang sama untuk validasi
                dist_matrix = torch.cdist(embeddings, embeddings, p=2.0)
                labels_equal = class_labels.view(-1, 1) == class_labels.view(1, -1)
                positive_mask = labels_equal & ~torch.eye(labels_equal.shape[0], dtype=torch.bool, device=device)
                negative_mask = ~labels_equal
                positive_dists = dist_matrix[positive_mask]
                negative_dists = dist_matrix[negative_mask]
                if len(positive_dists) > 0 and len(negative_dists) > 0:
                    loss_matrix = positive_dists.unsqueeze(1) - negative_dists.unsqueeze(0) + CONFIG["margin"]
                    loss = F.relu(loss_matrix).mean()
                else:
                    loss = torch.tensor(0.0, device=device)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Simpan Model Terbaik berdasarkan Val Loss Terendah
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(CONFIG["output_dir"], 'best_recognition_triplet_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f" -> Model terbaik disimpan di {save_path} (Val Loss: {best_val_loss:.4f})")

    print("\n--- Pelatihan Triplet Selesai ---")

if __name__ == '__main__':
    train_triplet()