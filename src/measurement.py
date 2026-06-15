"""
============================================================
SLV - SmartLivestock Vision
MODULE: measurement.py (PROPER EDITION)
============================================================
"""

import json
import math
import os

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    if not os.path.exists(_CONFIG_PATH):
        return {"mm_per_pixel": 0.3, "camera_index": 0}
    with open(_CONFIG_PATH, "r") as f:
        return json.load(f)

# ── Konversi Dimensi ─────────────────────────────────────

def pixels_to_mm(pixels: float) -> float:
    cfg = load_config()
    return pixels * cfg["mm_per_pixel"]

def toy_to_real_length(toy_mm: float) -> float:
    ratio = 150.0 / 31.0  # cm per mm mainan
    return toy_mm * ratio

def toy_to_real_height(toy_mm: float) -> float:
    ratio = 120.0 / 19.0  # cm per mm mainan
    return toy_mm * ratio

# ── Estimasi Lingkar Dada ─────────────────────────────────

def estimate_girth_cm(length_cm: float, height_cm: float) -> float:
    girth = 0.65 * length_cm + 0.55 * height_cm + 15.0
    return round(girth, 1)

# ── Estimasi Bobot ────────────────────────────────────────

def estimate_weight_schoorl(girth_cm: float) -> float:
    weight = ((girth_cm + 22) ** 2) / 100
    return round(weight, 1)

def estimate_weight_winter(length_cm: float, girth_cm: float) -> float:
    weight = (girth_cm ** 2 * length_cm) / 10840
    return round(weight, 1)

# ── Keseragaman (PROPER UPDATE - ROBUST STATS WITH ERROR HANDLING) ──

def calculate_uniformity(weights: list) -> dict:
    # Pengaman 1: Jika data kosong atau kurang dari 2 sapi, langsung tolak aman
    if not weights or len(weights) < 2:
        return {
            "mean": 0, "std": 0, "cv": 0,
            "uniformity": 0, "status": "⚠️ Butuh minimal 2 sapi untuk kalkulasi kelompok.",
            "min": 0, "max": 0, "count": len(weights) if weights else 0
        }

    try:
        n = len(weights)
        mean = sum(weights) / n
        variance = sum((w - mean) ** 2 for w in weights) / n
        std = math.sqrt(variance)
        
        # Pengaman 2: Hindari Division By Zero jika mean = 0
        cv = (std / mean * 100) if mean > 0 else 0
        
        # Batasi Uniformity Index agar tidak menghasilkan nilai minus (min=0, max=100)
        uniformity = max(0, min(100, 100 - cv * 2))

        if cv < 5:
            status = "🟢 Sangat Seragam (Excellent)"
        elif cv < 10:
            status = "🟡 Seragam (Good)"
        elif cv < 15:
            status = "🟠 Cukup Seragam (Fair)"
        else:
            status = "🔴 Tidak Seragam (Poor)"

        return {
            "mean"       : round(mean, 1),
            "std"        : round(std, 1),
            "cv"       : round(cv, 2),
            "uniformity" : round(uniformity, 1),
            "status"     : status,
            "min"        : round(min(weights), 1),
            "max"        : round(max(weights), 1),
            "count"      : n
        }
    except Exception as e:
        # Fallback jika ada error kalkulasi yang tidak terduga
        return {
            "mean": 0, "std": 0, "cv": 0,
            "uniformity": 0, "status": f"❌ Error perhitungan: {str(e)}",
            "min": 0, "max": 0, "count": len(weights)
        }

# ── Status Sapi ───────────────────────────────────────────

def get_livestock_status(weight_kg: float) -> dict:
    if weight_kg >= 400:
        return {
            "label": "✅ Siap Jual / Potong",
            "color": "#00C851",
            "recommendation": "Bobot optimal. Segera jadwalkan penjualan."
        }
    elif weight_kg >= 300:
        return {
            "label": "📈 Fase Penggemukan Akhir",
            "color": "#ffbb33",
            "recommendation": "Tingkatkan protein pakan. Target 2-3 bulan lagi."
        }
    elif weight_kg >= 200:
        return {
            "label": "🌱 Fase Penggemukan Awal",
            "color": "#33b5e5",
            "recommendation": "Pertumbuhan normal. Pantau konsistensi pakan."
        }
    else:
        return {
            "label": "⚠️ Perlu Perhatian Nutrisi",
            "color": "#FF4444",
            "recommendation": "Bobot di bawah standar. Cek kondisi kesehatan & pakan."
        }