import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------
# Bagian 1: Fungsi Helper & Modul PointNetSetAbstraction
# ---------------------------------------------------

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(xyz.device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, 3, device=xyz.device)
            grouped_xyz = xyz.view(xyz.shape[0], 1, xyz.shape[1], 3)
            if points is not None:
                grouped_points = points.view(points.shape[0], 1, points.shape[1], -1)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(xyz.shape[0], self.npoint, 1, 3)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        
        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)

        return new_xyz, new_points


# ---------------------------------------------------
# Bagian 2: Arsitektur Utama Model Recognition
# ---------------------------------------------------

class PointNet2Recognition(nn.Module):
    def __init__(self, embedding_dim=256):
        super(PointNet2Recognition, self).__init__()
        # in_channel = 6 if use_feature else 3
        in_channel = 3
        
        # Encoder (Backbone)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024], group_all=True)
        
        # Classifier Head
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        

    def forward(self, xyz):
        # xyz: [Batch Size, Jumlah Titik, 3]
        B, _, _ = xyz.shape
        
        # Fitur awal adalah koordinat itu sendiri (XYZ)
        # Untuk membuatnya 6 channel seperti model asli, bisa duplikasi
        # points = xyz.repeat(1, 1, 2) # [B, N, 6]
        
        # Encoder Path
        l1_xyz, l1_points = self.sa1(xyz, xyz)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # Lapisan terakhir mengambil semua fitur dan merangkumnya menjadi satu vektor global
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # x sekarang adalah vektor fitur global [B, 1024]
        x = l3_points.view(B, 1024)
        
        # Classifier Head
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        embedding = self.bn2(self.fc2(x))  # Embedding akhir
        # Outputnya adalah 'logits', yaitu skor mentah untuk setiap kelas
        # x = self.fc3(x)
        embedding = F.normalize(embedding, p=2, dim=1)  # Normalisasi embedding
        
        return embedding

# --- BLOK UNTUK PENGUJIAN MENGGUNAKAN DATASET ASLI ---
if __name__ == '__main__':
    from dataset import RugaeDataset
    from torch.utils.data import DataLoader

    print("--- Menjalankan Tes untuk Model Recognition dengan Dataset Asli ---")
    
    try:
        # Ganti 'test' dengan 'train' jika folder test Anda kosong
        train_dataset = RugaeDataset(split='train')
        
        if len(train_dataset) > 0:
            # num_unique_patients = len(train_dataset.classes)
            # print(f"Jumlah pasien unik (kelas) yang ditemukan: {num_unique_patients}")

            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
            first_batch = next(iter(train_loader))
            points_batch = first_batch['points']
            
            # Inisialisasi model dengan jumlah kelas yang benar
            embedding_dimension = 256 
            model = PointNet2Recognition(embedding_dim=embedding_dimension)
            
            print(f"Input shape (dari DataLoader): {points_batch.shape}")
            
            # Jalankan data batch melalui model
            embedding_output = model(points_batch)

            print(f"Output shape (prediksi/logits): {embedding_output.shape}")

            # Cek apakah output shape sesuai harapan
            # Misal, jika Batch Size = 2, dan ada 15 pasien unik
            # maka shape-nya harus (2, 15)
            expected_shape = (points_batch.shape[0], embedding_dimension)
            if embedding_output.shape == expected_shape:
                print("\n✅ Tes Berhasil! Output shape sesuai harapan.")
            else:
                print(f"\n❌ Tes Gagal! Output shape seharusnya {expected_shape}.")
        else:
            print("Dataset 'train' kosong. Tes tidak bisa dijalankan.")
            
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")