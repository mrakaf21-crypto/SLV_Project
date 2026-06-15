import os
from ultralytics import YOLO

def start_training():
    print("==========================================================")
    print("🚀 SCRIPT RE-TRAINING INSTAN YOLOv8 - SMARTLIVESTOCK VISION")
    print("==========================================================")
    
    # Memanggil bobot arsitektur dasar YOLOv8 Nano (Sangat direkomendasikan untuk komputasi lokal laptop)
    model = YOLO("yolov8n.pt")
    
    yaml_path = input("Masukkan lokasi berkas dataset.yaml (Contoh: dataset/data.yaml): ") or "dataset/data.yaml"
    
    if not os.path.exists(yaml_path):
        print(f"[⚠️ PERINGATAN] Berkas target: '{yaml_path}' tidak ditemukan di direktori ini!")
        
    epochs_input = int(input("Masukkan total interasi Epochs (Default 50): ") or 50)
    imgsz_input = int(input("Masukkan dimensi ukuran gambar input (Default 640): ") or 640)
    
    print("\n[PROSES] Menyalakan engine latihan jaringan syaraf tiruan...")
    
    model.train(
        data=yaml_path,
        epochs=epochs_input,
        imgsz=imgsz_input,
        plots=True
    )
    
    print("\n[SUKSES] Proses pembelajaran tuntas!")
    print("Bobot model kustom terbaik tersimpan di folder otomatis: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    start_training()