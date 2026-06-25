"""
model_service.py
────────────────
Adapted from main.py — exposes a single `predict_image()` function
for use by the Flask API without any CV2 display or CLI logic.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from ultralytics import YOLO

# ─── Paths (relative to ai_service/, model files sit one level up) ───────────
BASE_DIR         = Path(__file__).parent.parent
MODEL_PATH        = BASE_DIR / "best.pt"
NUTRITION_DB_PATH = BASE_DIR / "nutrition_database.json"

# Giảm ngưỡng tự tin xuống 0.35 để dễ dàng nhận diện khi có nhiều quả / quả nhỏ
CONFIDENCE_THRESHOLD = 0.35

CLASS_NAMES = [
    'apple', 'avocado', 'banana', 'dragon fruit',
    'lemon', 'mango', 'orange', 'papaya',
    'pineapple', 'strawberry'
]

CLASS_TO_NUTRITION_KEY = {
    'apple':        'apple_raw',
    'avocado':      'avocado_raw',
    'banana':       'banana_raw',
    'dragon fruit': 'dragonfruit_raw',
    'lemon':        'lemon_raw',
    'mango':        'mango_raw',
    'orange':       'orange_raw',
    'papaya':       'papaya_raw',
    'pineapple':    'pineapple_raw',
    'strawberry':   'strawberry_raw',
}

# ─── Ripeness rules (unchanged from main.py) ─────────────────────────────────
RIPENESS_RULES = {
    "apple": {
        "unripe":        {"hue": (25, 80),   "label": "Chưa chín", "stage": "unripe"},
        "ripe":          {"hue": (0,  10),   "label": "Đã chín",   "stage": "ripe"},
        "ripe_wrap":     (160, 180),
        "overripe":      {"hue": (0,   5),   "label": "Quá chín",  "stage": "overripe"},
        "overripe_wrap": (168, 180),
    },
    "avocado": {
        "unripe":   {"hue": (25, 90),   "label": "Chưa chín", "stage": "unripe"}, # Xanh lá
        "ripe":     {"hue": (120, 180), "label": "Đã chín",   "stage": "ripe"},   # Tím/Đen
        "ripe_wrap": (0, 20),                                                     # Nâu sẫm
        "overripe": {"hue": (0,   5),   "label": "Quá chín",  "stage": "overripe"},
    },
    "banana": {
        "unripe":   {"hue": (28, 85),  "label": "Chưa chín", "stage": "unripe"},
        "ripe":     {"hue": (15, 30),  "label": "Đã chín",   "stage": "ripe"},
        "overripe": {"hue": (0,  17),  "label": "Quá chín",  "stage": "overripe"},
    },
    "dragon fruit": {
        "unripe":   {"hue": (28, 80),   "label": "Chưa chín", "stage": "unripe"},
        "ripe":     {"hue": (140, 170), "label": "Đã chín",   "stage": "ripe"},
        "overripe": {"hue": (0,   10),  "label": "Quá chín",  "stage": "overripe"},
    },
    "lemon": {
        "unripe":   {"hue": (45, 90),  "label": "Chưa chín", "stage": "unripe"},
        "ripe":     {"hue": (30, 45),  "label": "Đã chín",   "stage": "ripe"},
        "overripe": {"hue": (0,  30),  "label": "Quá chín",  "stage": "overripe"},
    },
    "mango": {
        "unripe":   {"hue": (32, 85),  "label": "Chưa chín", "stage": "unripe"},
        "ripe":     {"hue": (13, 35),  "label": "Đã chín",   "stage": "ripe"},
        "overripe": {"hue": (0,  15),  "label": "Quá chín",  "stage": "overripe"},
    },
    "orange": {
        "unripe":        {"hue": (28, 80),  "label": "Chưa chín", "stage": "unripe"},
        "ripe":          {"hue": (8,  28),  "label": "Đã chín",   "stage": "ripe"},
        "overripe":      {"hue": (0,  10),  "label": "Quá chín",  "stage": "overripe"},
        "overripe_wrap": (168, 180),
    },
    "papaya": {
        "unripe":   {"hue": (28, 80),  "label": "Chưa chín", "stage": "unripe"},
        "ripe":     {"hue": (8,  25),  "label": "Đã chín",   "stage": "ripe"},
        "overripe": {"hue": (0,  10),  "label": "Quá chín",  "stage": "overripe"},
    },
    "pineapple": {
        "unripe":   {"hue": (28, 80),  "label": "Chưa chín", "stage": "unripe"},
        "ripe":     {"hue": (16, 30),  "label": "Đã chín",   "stage": "ripe"},
        "overripe": {"hue": (0,  18),  "label": "Quá chín",  "stage": "overripe"},
    },
    "strawberry": {
        "unripe":        {"hue": (28, 80),  "label": "Chưa chín", "stage": "unripe"},
        "ripe":          {"hue": (0,  10),  "label": "Đã chín",   "stage": "ripe"},
        "ripe_wrap":     (160, 180),
        "overripe":      {"hue": (0,   5),  "label": "Quá chín",  "stage": "overripe"},
        "overripe_wrap": (170, 180),
    },
}

SAT_MIN          = 40
VAL_MIN          = 40
VAL_MAX          = 235
MIN_VALID_PIXELS = 50


@dataclass
class RipenessResult:
    stage:        str
    label:        str
    dominant_hue: float
    score:        float


class RipenessAnalyzer:
    def __init__(self, roi_shrink: float = 0.15):
        self.roi_shrink = roi_shrink

    def analyze(self, frame: np.ndarray, bbox: list, class_name: str) -> RipenessResult:
        roi = self._crop_roi(frame, bbox)
        if roi is None or roi.size == 0:
            return self._unknown()
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = self._valid_pixel_mask(hsv)
        if cv2.countNonZero(mask) < MIN_VALID_PIXELS:
            return self._unknown()
        dominant_hue = self._dominant_hue(hsv[:, :, 0], mask)
        return self._classify(class_name.lower(), dominant_hue)

    def _crop_roi(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h_f, w_f = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_f, x2); y2 = min(h_f, y2)
        w = x2 - x1; h = y2 - y1
        if w < 10 or h < 10:
            return None
        px = int(w * self.roi_shrink)
        py = int(h * self.roi_shrink)
        ix1, iy1 = x1 + px, y1 + py
        ix2, iy2 = x2 - px, y2 - py
        if ix2 <= ix1 or iy2 <= iy1:
            return frame[y1:y2, x1:x2]
        return frame[iy1:iy2, ix1:ix2]

    def _valid_pixel_mask(self, hsv):
        s = hsv[:, :, 1]; v = hsv[:, :, 2]
        mask = (s >= SAT_MIN) & (v >= VAL_MIN) & (v <= VAL_MAX)
        return mask.astype(np.uint8) * 255

    def _dominant_hue(self, hue_ch, mask):
        hist = cv2.calcHist([hue_ch], [0], mask, [36], [0, 180])
        peak_bin = int(np.argmax(hist.flatten()))
        return peak_bin * 5.0 + 2.5

    def _classify(self, class_name, hue):
        rules = RIPENESS_RULES.get(class_name)
        if rules is None:
            return self._unknown()

        def in_range(h, lo, hi): return lo <= h <= hi

        o = rules.get("overripe", {})
        o_h = o.get("hue", (999, 999))
        o_w = rules.get("overripe_wrap")
        if in_range(hue, *o_h) or (o_w and in_range(hue, *o_w)):
            return RipenessResult(o.get("stage", "overripe"), o.get("label", "Quá chín"), hue, 0.85)

        r = rules.get("ripe", {})
        r_h = r.get("hue", (999, 999))
        r_w = rules.get("ripe_wrap")
        if in_range(hue, *r_h) or (r_w and in_range(hue, *r_w)):
            return RipenessResult(r.get("stage", "ripe"), r.get("label", "Đã chín"), hue, 0.90)

        u = rules.get("unripe", {})
        return RipenessResult(u.get("stage", "unripe"), u.get("label", "Chưa chín"), hue, 0.80)

    def _unknown(self):
        return RipenessResult("unknown", "Không rõ", -1.0, 0.0)


# ─── Singletons (loaded once at startup) ─────────────────────────────────────
_model: Optional[YOLO] = None
_nutrition_db: dict = {}
_analyzer = RipenessAnalyzer(roi_shrink=0.15)


def load_resources():
    global _model, _nutrition_db
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    print(f"[AI] Loading model from {MODEL_PATH} ...")
    _model = YOLO(str(MODEL_PATH))
    print("[AI] ✓ Model loaded.")

    if NUTRITION_DB_PATH.exists():
        with open(NUTRITION_DB_PATH, "r", encoding="utf-8") as f:
            _nutrition_db = json.load(f)
        print(f"[AI] ✓ Loaded {len(_nutrition_db)} nutrition entries.")
    else:
        print("[AI] ⚠ Nutrition DB not found.")


def predict_image(image_bytes: bytes) -> dict:
    """
    Accept raw image bytes.
    Returns a dict with prediction results, or raises ValueError if no fruit found.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_resources() first.")

    # Decode image
    nparr  = np.frombuffer(image_bytes, np.uint8)
    image  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image. Invalid format.")

    results = _model.predict(source=image, conf=CONFIDENCE_THRESHOLD, iou=0.45, augment=True, verbose=False)

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cid = int(box.cls[0])
            if cid >= len(CLASS_NAMES):
                continue
            conf   = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            detections.append((cid, conf, coords))

    if not detections:
        return {"success": False, "message": "No fruit detected in the image."}

    # Pick highest-confidence detection
    detections.sort(key=lambda x: x[1], reverse=True)
    cid, conf, coords = detections[0]
    class_name = CLASS_NAMES[cid]

    # --- Heuristic: Fix Banana vs Mango confusion based on aspect ratio ---
    x1, y1, x2, y2 = coords
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect_ratio = max(w / h, h / w)
    
    # Lấy màu sắc chủ đạo (Hue) trước để giúp tinh chỉnh kết quả
    temp_ripeness = _analyzer.analyze(image, coords, class_name)
    hue = temp_ripeness.dominant_hue

    # Chuối đơn lẻ thường dài (aspect ratio > 2.0). 
    # Nếu AI nhận nhầm 1 quả chuối dài thành xoài, ta sửa lại thành chuối.
    if class_name == 'mango' and aspect_ratio >= 2.0:
        class_name = 'banana'
        
    # Nếu AI nhầm thành táo:
    if class_name == 'apple':
        if aspect_ratio >= 1.2:
            # Táo thường rất tròn. Nếu thuôn dài (aspect ratio > 1.2):
            if 10 <= hue <= 35:
                if aspect_ratio >= 1.3:
                    class_name = 'mango'  # Vàng + Dài -> Xoài
            else:
                class_name = 'avocado'    # Màu xanh/đậm + Bầu dục -> Bơ
        else:
            # Hình tròn (aspect ratio < 1.2) giống táo, nhưng Táo thường màu đỏ (0-10) hoặc xanh (30-80).
            # Rất hiếm táo có màu cam rực (12-25) ở Việt Nam. Nếu tròn + màu cam rực -> Quả Cam.
            if 12 <= hue <= 26:
                class_name = 'orange'
    # Không có quả bơ nào màu vàng! Nếu AI nhận là bơ nhưng màu lại vàng/cam rực rỡ (Hue 10-30):
    if class_name == 'avocado' and 10 <= hue <= 30:
        # Nếu hình dáng dài thì đổi thành xoài, nếu tròn/ngắn thì đổi thành cam (orange)
        if aspect_ratio >= 1.3:
            class_name = 'mango'
        else:
            class_name = 'orange'
    # ----------------------------------------------------------------------

    # Ripeness
    ripeness = _analyzer.analyze(image, coords, class_name)

    # Nutrition
    key = CLASS_TO_NUTRITION_KEY.get(class_name.lower())
    nutrition = _nutrition_db.get(key) if key else None

    result = {
        "success":     True,
        "fruit_name":  class_name,
        "confidence":  round(conf * 100, 2),
        "ripeness":    ripeness.stage,
        "ripeness_label": ripeness.label,
        "all_detections": [
            {
                "fruit_name": CLASS_NAMES[d[0]],
                "confidence": round(d[1] * 100, 2)
            }
            for d in detections
        ]
    }

    if nutrition:
        result["nutrition"] = {
            "name_en":       nutrition.get("name_en", class_name),
            "name_vn":       nutrition.get("name_vn", ""),
            "energy_kcal":   nutrition.get("energy_kcal"),
            "protein_g":     nutrition.get("protein_g"),
            "fat_g":         nutrition.get("fat_g"),
            "carbs_g":       nutrition.get("carbs_g"),
            "fiber_g":       nutrition.get("fiber_g"),
            "sugar_g":       nutrition.get("sugar_g"),
            "calcium_mg":    nutrition.get("calcium_mg"),
            "iron_mg":       nutrition.get("iron_mg"),
            "magnesium_mg":  nutrition.get("magnesium_mg"),
            "potassium_mg":  nutrition.get("potassium_mg"),
            "zinc_mg":       nutrition.get("zinc_mg"),
            "vitamin_c_mg":  nutrition.get("vitamin_c_mg"),
            "vitamin_a_iu":  nutrition.get("vitamin_a_iu"),
        }
    else:
        result["nutrition"] = None

    return result
