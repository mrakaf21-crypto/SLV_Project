import json
import os

def run_calibration():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    
    print("==========================================================")
    print("📐 SISTEM CONFIG KALIBRASI SPASIAL - SMARTLIVESTOCK VISION")
    print("==========================================================")
    
    try:
        mm_per_pixel = float(input("Masukkan nilai mm/pixel aktual (Default 0.27): ") or 0.27)
        scale_length = float(input("Masukkan nilai Skala Panjang P (Default 48.3871): ") or 48.3871)
        scale_height = float(input("Masukkan nilai Skala Tinggi T (Default 63.1579): ") or 63.1579)
        
        config_data = {
            "mm_per_pixel": mm_per_pixel,
            "scale_length": scale_length,
            "scale_height": scale_height
        }
        
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)
            
        print("\n[SUKSES] Parameter kalibrasi mutakhir berhasil disimpan!")
        print(f"Path berkas: {config_path}")
        print(json.dumps(config_data, indent=4))
        
    except ValueError:
        print("\n[EROR] Input gagal. Pastikan memasukkan format angka desimal murni!")

if __name__ == "__main__":
    run_calibration()