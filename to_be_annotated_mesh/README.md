struktur directory :

* folder anotated\_dataset > subfolder dataset (langsung ke identitas data, tidak didahului folder lain)
* struktur dataset lihat pada bagian dibawah





/proyek-rugae/
|
|-- 📄 preprocess\_initial.py
|-- 📄 preprocess\_final.py  
|
|-- 📂 dataset/                  # INPUT: Folder data mentah (.stl, .ply, .obj)
|   |-- 📂 K01A (Bernado Barus)/  # PENTING: subfolder dari dataset berupa identitas ID-Pasien(Nama Pasien)
|   |   |-- ...-upperjaw.ply      # di dalamnya langsung file scan 3d image dengan ekstensi .ply dll
|   |
|-- 📂 to\_be\_annotated\_mesh/      # OUTPUT 1: Data bersih untuk anotator
|   |-- 📂 K01A (Bernado Barus)/
|   |   |-- ...\_cleaned.ply
|   |
|-- 📂 annotated\_dataset/         # INPUT 2: Hasil kerja anotator
|   |-- 📂 K01A (Bernado Barus)/
|   |   |-- ...\_landmarks.txt
|   |   |-- ...\_annotated.ply
|   |
|-- 📂 final\_dataset\_for\_training/ # OUTPUT FINAL: Data siap latih
|-- 📂 train/
|   |-- ...\_final.npz
|-- 📂 val/
|   |-- ...\_final.npz
|-- 📂 test/
    |-- ...\_final.npz

