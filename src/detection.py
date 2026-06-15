"""
============================================================
SLV - SmartLivestock Vision
MODULE: detection.py (COMPACT HUD LABELS OPTIMIZED)
============================================================
Modul deteksi objek menggunakan YOLOv8 dengan HUD super ringkas.
============================================================
"""

import cv2
import os
import json
import numpy as np
import streamlit as st

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics tidak terinstall. Jalankan: pip install ultralytics")

from measurement import (
    pixels_to_mm, toy_to_real_length, toy_to_real_height,
    estimate_girth_cm, estimate_weight_schoorl,
    estimate_weight_winter, get_livestock_status
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
CUSTOM_MODEL = os.path.join(MODEL_DIR, "best.pt")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"mm_per_pixel": 0.15, "camera_index": 0}

COLORS = [
    (0, 255, 0),    # hijau
    (0, 200, 255),  # kuning
    (255, 100, 0),  # biru muda
    (180, 0, 255),  # ungu/pink
    (0, 255, 180),  # cyan
]

class LivestockDetector:
    def __init__(self):
        self.model  = None
        self.mode   = "contour"
        self.config = load_config()
        self.active_id_map = {} 
        self._load_model()

    def _load_model(self):
        if not YOLO_AVAILABLE:
            self.mode = "contour"
            return

        if os.path.exists(CUSTOM_MODEL):
            self.model = YOLO(CUSTOM_MODEL)
            self.mode  = "custom"
        else:
            try:
                self.model = YOLO("yolov8n.pt")
                self.mode  = "pretrained"
            except Exception:
                self.mode = "contour"

    def detect(self, frame: np.ndarray, selected_id: int = None) -> dict:
        if self.mode in ("custom", "pretrained"):
            return self._detect_yolo(frame, selected_id)
        else:
            return self._detect_contour(frame, selected_id)

    def _draw_smart_label(self, frame, x1, y1, display_id, conf, detection, color, is_sel):
        """
        Fungsi HUD Label Multi-Line yang sudah diperkecil dan dirapatkan kotaknya
        """
        conf_pct = int(conf * 100)
        p_cm = detection['panjang_cm']
        t_cm = detection['tinggi_cm']
        bobot = detection['bobot_kg']

        line1 = f"ID-{display_id} ({conf_pct}%)"
        line2 = f"P: {p_cm}cm | T: {t_cm}cm"
        line3 = f"Bobot: {bobot}kg"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38  # OPTIMASI: Dikecilkan dari 0.5 agar proporsional
        thickness = 1

        # Hitung dimensi teks aktual
        (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
        (w3, h3), _ = cv2.getTextSize(line3, font, font_scale, thickness)

        max_w = max(w1, w2, w3)
        
        # OPTIMASI: Set margins & spasi super rapat (Compact Padding)
        pad_x = 5
        pad_y = 3
        line_spacing = 3

        total_h = h1 + h2 + h3 + (line_spacing * 2) + (pad_y * 2)

        # Geser posisi kotak sedikit ke atas bounding box asli
        bg_y1 = max(0, y1 - total_h - 3)
        bg_y2 = bg_y1 + total_h
        bg_x2 = x1 + max_w + (pad_x * 2)

        # Gambar Kotak Background Solid mengikuti teks yang mengecil
        cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), color, -1) 
        
        if is_sel:
            cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)

        text_color = (0, 0, 0)
        
        # Render Baris 1 dengan Anti-Aliasing (cv2.LINE_AA) agar tulisan kecil tidak pecah
        y_pos = bg_y1 + pad_y + h1
        cv2.putText(frame, line1, (x1 + pad_x, y_pos), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Render Baris 2
        y_pos += h2 + line_spacing
        cv2.putText(frame, line2, (x1 + pad_x, y_pos), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Render Baris 3
        y_pos += h3 + line_spacing
        cv2.putText(frame, line3, (x1 + pad_x, y_pos), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def _detect_yolo(self, frame, selected_id):
        annotated = frame.copy()
        detections = []

        h, w = frame.shape[:2]
        input_frame = cv2.resize(frame, (640, int(640 * h / w)))

        if self.mode == "pretrained":
            results = self.model.track(input_frame, classes=[19], persist=True, tracker="bytetrack.yaml", verbose=False)
        else:
            results = self.model.track(input_frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        scale_x = w / 640
        scale_y = h / (640 * h / w)
        
        current_raw_ids = []
        valid_boxes = []

        for box in results[0].boxes:
            conf = float(box.conf[0])
            try:
                active_conf_threshold = st.session_state.get("ai_conf", 0.60)
            except Exception:
                active_conf_threshold = 0.60
            
            if conf < active_conf_threshold:  
                continue

            raw_id = int(box.id[0]) if box.id is not None else -1
            valid_boxes.append((raw_id, box, conf))
            
            if raw_id != -1:
                current_raw_ids.append(raw_id)

        keys_to_delete = [k for k in self.active_id_map.keys() if k not in current_raw_ids]
        for k in keys_to_delete:
            del self.active_id_map[k]

        for raw_id, box, conf in valid_boxes:
            if raw_id == -1:
                display_id = len(self.active_id_map) + 1
            else:
                if raw_id not in self.active_id_map:
                    used_ids = set(self.active_id_map.values())
                    new_id = 1
                    while new_id in used_ids:
                        new_id += 1
                    self.active_id_map[raw_id] = new_id
                display_id = self.active_id_map[raw_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)

            detection = self._calc_metrics(display_id, x1, y1, x2, y2, conf)
            if detection is None:
                continue

            detections.append(detection)
            color = COLORS[display_id % len(COLORS)]
            is_sel = (selected_id == display_id)

            # Bounding Box Sapi (Bersih tanpa bulatan tengah)
            thickness = 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Gambar HUD Label mini yang pas menyesuaikan kotak sapi lo
            self._draw_smart_label(annotated, x1, y1, display_id, conf, detection, color, is_sel)

            if is_sel:
                self._draw_dimension_lines(annotated, x1, y1, x2, y2, color)

        return {"detections": detections, "annotated": annotated}

    def _detect_contour(self, frame, selected_id):
        annotated = frame.copy()
        detections = []

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (frame.shape[0] * frame.shape[1]) * 0.002
        valid = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) > min_area]
        valid.sort(key=lambda x: -x[0])
        valid = valid[:5]

        cow_counter = 1
        for area, cnt in valid:
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = x, y, x + w, y + h

            detection = self._calc_metrics(cow_counter, x1, y1, x2, y2, conf=0.85)
            if detection is None:
                continue

            detections.append(detection)
            color  = COLORS[cow_counter % len(COLORS)]
            is_sel = (selected_id == cow_counter)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            self._draw_smart_label(annotated, x1, y1, cow_counter, 0.85, detection, color, is_sel)

            if is_sel:
                self._draw_dimension_lines(annotated, x1, y1, x2, y2, color)
            
            cow_counter += 1

        cv2.putText(annotated, "Mode: OpenCV Contour (Tanpa YOLO)", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
        return {"detections": detections, "annotated": annotated}

    def _calc_metrics(self, cow_id, x1, y1, x2, y2, conf):
        w_px = x2 - x1
        h_px = y2 - y1

        toy_p_mm = pixels_to_mm(w_px)
        toy_t_mm = pixels_to_mm(h_px)

        real_p_cm = toy_to_real_length(toy_p_mm)
        real_t_cm = toy_to_real_height(toy_t_mm)
        
        if real_p_cm > 280.0 or real_p_cm < 50.0 or real_t_cm > 200.0 or real_t_cm < 40.0:
            return None

        girth_cm  = estimate_girth_cm(real_p_cm, real_t_cm)
        weight_s  = estimate_weight_schoorl(girth_cm)
        weight_w  = estimate_weight_winter(real_p_cm, girth_cm)
        weight    = (weight_s + weight_w) / 2
        status    = get_livestock_status(weight)

        return {
            "id"          : cow_id,
            "bbox"        : (x1, y1, x2, y2),
            "confidence"  : round(conf, 2),
            "toy_panjang" : round(toy_p_mm, 1),
            "toy_tinggi"  : round(toy_t_mm, 1),
            "panjang_cm"  : round(real_p_cm, 1),
            "tinggi_cm"   : round(real_t_cm, 1),
            "lingkar_cm"  : girth_cm,
            "bobot_kg"    : round(weight, 1),
            "status"      : status,
        }

    def _draw_dimension_lines(self, frame, x1, y1, x2, y2, color):
        cy = (y1 + y2) // 2
        cv2.arrowedLine(frame, (x1, cy), (x2, cy), (255, 255, 255), 1, tipLength=0.05)
        cx = (x1 + x2) // 2
        cv2.arrowedLine(frame, (cx, y1), (cx, y2), (255, 255, 255), 1, tipLength=0.05)