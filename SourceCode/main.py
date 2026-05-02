"""
============================================================
  Nhận diện trái cây + Đánh giá độ chín bằng HSV
  Camera: App "IP Webcam" (Android) stream qua WiFi
  Model:  YOLOv8 (best.pt)

  Cách dùng:
    1. Cài app IP Webcam trên điện thoại Android
    2. Mở app → kéo xuống → bấm "Start server"
    3. App hiện địa chỉ VD: http://192.168.1.5:8080
    4. Điền IP đó vào PHONE_IP bên dưới
    5. python main.py
============================================================
"""

# ──────────────────────────────────────────────
# BƯỚC 0: Import thư viện
# ──────────────────────────────────────────────
import cv2
import sys
import time
import urllib.request
import numpy as np
from pathlib import Path
from collections import deque, Counter
from dataclasses import dataclass
from typing import Optional
from ultralytics import YOLO


# ══════════════════════════════════════════════
#  PHẦN 1: CẤU HÌNH CHUNG
# ══════════════════════════════════════════════

# ⚠️ Điền IP điện thoại hiển thị trong app IP Webcam
PHONE_IP   = "192.168.2.106"
PHONE_PORT = 8080
STREAM_URL = f"http://{PHONE_IP}:{PHONE_PORT}/video"

MODEL_PATH           = "best.pt"
CONFIDENCE_THRESHOLD = 0.60
WINDOW_SIZE          = 6       # Số frame cho temporal smoothing
VOTE_RATIO           = 0.55    # Tỉ lệ vote tối thiểu để công nhận nhãn

CLASS_NAMES = [
    'apple', 'avocado', 'banana', 'dragon fruit',
    'lemon', 'mango', 'orange', 'papaya',
    'pineapple', 'strawberry'
]

# Màu bounding box BGR theo từng lớp
CLASS_COLORS_BGR = [
    (0,   51,  255),   # apple        → Đỏ
    (51,  170,  51),   # avocado      → Xanh lá
    (0,   215, 255),   # banana       → Vàng
    (204,  51, 204),   # dragon fruit → Tím
    (68,  255, 255),   # lemon        → Vàng chanh
    (0,   153, 255),   # mango        → Cam
    (0,   102, 255),   # orange       → Cam đậm
    (119, 187, 255),   # papaya       → Cam nhạt
    (136, 255,  68),   # pineapple    → Xanh vàng
    (153,  51, 255),   # strawberry   → Hồng
]


# ══════════════════════════════════════════════
#  PHẦN 2: MODULE PHÂN TÍCH ĐỘ CHÍN (HSV)
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# Bảng ngưỡng Hue per-fruit
# Hue trong OpenCV: 0–180 (bằng nửa góc thực 0–360°)
# Mỗi mức: hue=(lo, hi), label tiếng Việt, màu badge BGR
# ──────────────────────────────────────────────
RIPENESS_RULES = {
    "apple": {
        "unripe":        {"hue": (25, 80),   "label": "Chưa chín 🟢", "color_bgr": (50,  180,  50)},
        "ripe":          {"hue": (0,  10),   "label": "Đã chín  ✅",  "color_bgr": (30,   50, 210)},
        "ripe_wrap":     (160, 180),          # Đỏ táo nằm ở cả đầu cao Hue
        "overripe":      {"hue": (0,   5),   "label": "Quá chín 🟤", "color_bgr": (20,   20, 120)},
        "overripe_wrap": (168, 180),
    },
    "avocado": {
        "unripe":   {"hue": (28, 90),  "label": "Chưa chín 🟢", "color_bgr": (50, 180,  50)},
        "ripe":     {"hue": (10, 32),  "label": "Đã chín  ✅",  "color_bgr": (40, 110,  60)},
        "overripe": {"hue": (0,  12),  "label": "Quá chín 🟤", "color_bgr": (20,  20, 100)},
    },
    "banana": {
        "unripe":   {"hue": (28, 85),  "label": "Chưa chín 🟢", "color_bgr": (50, 180,  50)},
        "ripe":     {"hue": (15, 30),  "label": "Đã chín  ✅",  "color_bgr": (0,  220, 255)},
        "overripe": {"hue": (0,  17),  "label": "Quá chín 🟤", "color_bgr": (30,  30, 160)},
    },
    "dragon fruit": {
        "unripe":   {"hue": (28, 80),   "label": "Chưa chín 🟢", "color_bgr": (50, 180,  50)},
        "ripe":     {"hue": (140, 170), "label": "Đã chín  ✅",  "color_bgr": (180, 50, 180)},
        "overripe": {"hue": (0,   10),  "label": "Quá chín 🟤", "color_bgr": (30,  30, 140)},
    },
    "lemon": {
        "unripe":   {"hue": (28, 85),  "label": "Chưa chín 🟢", "color_bgr": (50, 180,  50)},
        "ripe":     {"hue": (20, 30),  "label": "Đã chín  ✅",  "color_bgr": (0,  240, 240)},
        "overripe": {"hue": (0,  22),  "label": "Quá chín 🟤", "color_bgr": (30,  30, 150)},
    },
    "mango": {
        "unripe":   {"hue": (32, 85),  "label": "Chưa chín 🟢", "color_bgr": (50, 180,  50)},
        "ripe":     {"hue": (13, 35),  "label": "Đã chín  ✅",  "color_bgr": (0,  200, 255)},
        "overripe": {"hue": (0,  15),  "label": "Quá chín 🟤", "color_bgr": (30,  30, 170)},
    },
    "orange": {
        "unripe":        {"hue": (28, 80),  "label": "Chưa chín 🟢", "color_bgr": (50, 180,  50)},
        "ripe":          {"hue": (8,  28),  "label": "Đã chín  ✅",  "color_bgr": (0,  165, 255)},
        "overripe":      {"hue": (0,  10),  "label": "Quá chín 🟤", "color_bgr": (30,  30, 160)},
        "overripe_wrap": (168, 180),
    },
    "papaya": {
        "unripe":   {"hue": (28, 80),  "label": "Chưa chín 🟢", "color_bgr": (50, 180,  50)},
        "ripe":     {"hue": (8,  25),  "label": "Đã chín  ✅",  "color_bgr": (30, 160, 255)},
        "overripe": {"hue": (0,  10),  "label": "Quá chín 🟤", "color_bgr": (30,  30, 160)},
    },
    "pineapple": {
        "unripe":   {"hue": (28, 80),  "label": "Chưa chín 🟢", "color_bgr": (50, 180,  50)},
        "ripe":     {"hue": (16, 30),  "label": "Đã chín  ✅",  "color_bgr": (0,  210, 255)},
        "overripe": {"hue": (0,  18),  "label": "Quá chín 🟤", "color_bgr": (30,  30, 150)},
    },
    "strawberry": {
        "unripe":        {"hue": (28, 80),  "label": "Chưa chín 🟢", "color_bgr": (50, 180,  50)},
        "ripe":          {"hue": (0,  10),  "label": "Đã chín  ✅",  "color_bgr": (50,  50, 210)},
        "ripe_wrap":     (160, 180),
        "overripe":      {"hue": (0,   5),  "label": "Quá chín 🟤", "color_bgr": (20,  20, 120)},
        "overripe_wrap": (170, 180),
    },
}

# Ngưỡng lọc pixel nhiễu
SAT_MIN          = 40    # Loại pixel xám / trắng nhạt
VAL_MIN          = 40    # Loại pixel tối / bóng
VAL_MAX          = 235   # Loại pixel trắng bão hòa
MIN_VALID_PIXELS = 50    # Số pixel hợp lệ tối thiểu để phân tích


@dataclass
class RipenessResult:
    stage:        str    # "unripe" | "ripe" | "overripe" | "unknown"
    label_vi:     str    # Nhãn hiển thị tiếng Việt
    color_bgr:    tuple  # Màu BGR cho badge
    dominant_hue: float  # Hue trội nhất (để debug)
    score:        float  # Độ tin cậy 0.0–1.0


class RipenessAnalyzer:
    """
    Phân tích độ chín trái cây bằng cách so sánh phân bố
    Hue trong không gian màu HSV với ngưỡng per-fruit.

    Quy trình:
        1. Crop ROI từ bounding box (thu nhỏ vào trong để tránh background)
        2. Chuyển BGR → HSV
        3. Lọc pixel nhiễu bằng mask Saturation & Value
        4. Tính histogram Hue 36 bins → tìm dominant hue
        5. So sánh ngưỡng → trả RipenessResult
    """

    def __init__(self, roi_shrink: float = 0.15):
        """
        Args:
            roi_shrink: Tỉ lệ thu nhỏ bbox (0.15 = bỏ 15% mỗi cạnh).
                        Tăng nếu nền hay lẫn vào ROI.
        """
        self.roi_shrink = roi_shrink

    # ── Public API ────────────────────────────

    def analyze(
        self,
        frame: np.ndarray,
        bbox: list,
        class_name: str
    ) -> RipenessResult:
        """
        Phân tích một trái cây.

        Args:
            frame:      Frame BGR đầy đủ từ camera
            bbox:       [x1, y1, x2, y2] tọa độ pixel
            class_name: Tên lớp YOLO (VD: "banana")

        Returns:
            RipenessResult
        """
        roi = self._crop_roi(frame, bbox)
        if roi is None or roi.size == 0:
            return self._unknown()

        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = self._valid_pixel_mask(hsv)

        if cv2.countNonZero(mask) < MIN_VALID_PIXELS:
            return self._unknown()

        dominant_hue = self._dominant_hue(hsv[:, :, 0], mask)
        return self._classify(class_name.lower(), dominant_hue)

    def draw_badge(
        self,
        frame: np.ndarray,
        bbox: list,
        result: RipenessResult
    ) -> None:
        """
        Vẽ badge độ chín ngay bên dưới bounding box.
        Gọi SAU draw_detection() để tránh đè lên nhau.

        Args:
            frame:  Frame đang vẽ (in-place)
            bbox:   [x1, y1, x2, y2]
            result: Kết quả từ analyze()
        """
        if result.stage == "unknown":
            return

        x1, _, _, y2 = map(int, bbox)
        label = f"  {result.label_vi}  "
        fs, th = 0.55, 1

        (tw, txh), bl = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, fs, th
        )
        # Đặt badge ngay dưới thanh vote (cách y2 + 18px)
        badge_y = y2 + 26

        cv2.rectangle(
            frame,
            (x1, badge_y - txh - 4),
            (x1 + tw + 4, badge_y + bl),
            result.color_bgr,
            cv2.FILLED
        )
        cv2.putText(
            frame, label,
            (x1 + 2, badge_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, fs,
            (255, 255, 255), th, cv2.LINE_AA
        )

    # ── Private helpers ───────────────────────

    def _crop_roi(self, frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """Crop ROI, thu nhỏ vào trong roi_shrink% mỗi cạnh."""
        x1, y1, x2, y2 = map(int, bbox)
        h_f, w_f = frame.shape[:2]

        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(w_f, x2); y2 = min(h_f, y2)

        w = x2 - x1;  h = y2 - y1
        if w < 10 or h < 10:
            return None

        px = int(w * self.roi_shrink)
        py = int(h * self.roi_shrink)
        ix1, iy1 = x1 + px, y1 + py
        ix2, iy2 = x2 - px, y2 - py

        if ix2 <= ix1 or iy2 <= iy1:
            return frame[y1:y2, x1:x2]

        return frame[iy1:iy2, ix1:ix2]

    def _valid_pixel_mask(self, hsv: np.ndarray) -> np.ndarray:
        """
        Mask loại bỏ pixel bóng (V thấp), pixel trắng (S thấp),
        và pixel cháy sáng (V cao). Giữ lại màu sắc thực của vỏ.
        """
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        mask = (s >= SAT_MIN) & (v >= VAL_MIN) & (v <= VAL_MAX)
        return mask.astype(np.uint8) * 255

    def _dominant_hue(self, hue_ch: np.ndarray, mask: np.ndarray) -> float:
        """
        Tính dominant Hue bằng histogram 36 bins (mỗi bin = 5°).
        Trả về góc Hue 0–180 của đỉnh histogram.
        """
        hist = cv2.calcHist([hue_ch], [0], mask, [36], [0, 180])
        hist = hist.flatten()
        peak_bin = int(np.argmax(hist))
        return peak_bin * 5.0 + 2.5   # Trung tâm bin

    def _classify(self, class_name: str, hue: float) -> RipenessResult:
        """So khớp dominant_hue với bảng ngưỡng per-fruit."""
        rules = RIPENESS_RULES.get(class_name)
        if rules is None:
            return self._unknown()

        def in_range(h, lo, hi):
            return lo <= h <= hi

        # Kiểm tra overripe trước (ngưỡng hẹp nhất)
        o = rules.get("overripe", {})
        o_h = o.get("hue", (999, 999))
        o_w = rules.get("overripe_wrap")
        if in_range(hue, *o_h) or (o_w and in_range(hue, *o_w)):
            return RipenessResult("overripe", o.get("label", "Quá chín 🟤"),
                                  o.get("color_bgr", (30, 30, 160)), hue, 0.85)

        # Kiểm tra ripe
        r = rules.get("ripe", {})
        r_h = r.get("hue", (999, 999))
        r_w = rules.get("ripe_wrap")
        if in_range(hue, *r_h) or (r_w and in_range(hue, *r_w)):
            return RipenessResult("ripe", r.get("label", "Đã chín  ✅"),
                                  r.get("color_bgr", (0, 180, 80)), hue, 0.90)

        # Mặc định: unripe
        u = rules.get("unripe", {})
        return RipenessResult("unripe", u.get("label", "Chưa chín 🟢"),
                              u.get("color_bgr", (50, 180, 50)), hue, 0.80)

    def _unknown(self) -> RipenessResult:
        return RipenessResult("unknown", "Không rõ", (100, 100, 100), -1.0, 0.0)


# ══════════════════════════════════════════════
#  PHẦN 3: LOAD MODEL
# ══════════════════════════════════════════════

def load_model(model_path: str) -> YOLO:
    """Load file best.pt với kiểm tra lỗi."""
    if not Path(model_path).exists():
        print(f"[LỖI] Không tìm thấy '{model_path}'")
        sys.exit(1)
    print("[INFO] Đang load model ...")
    model = YOLO(model_path)
    print("[INFO] ✓ Load model thành công!")
    return model


# ══════════════════════════════════════════════
#  PHẦN 4: KẾT NỐI IP WEBCAM
# ══════════════════════════════════════════════

def open_ip_camera(stream_url: str) -> cv2.VideoCapture:
    """
    Kết nối tới stream MJPEG từ app IP Webcam.
    Kiểm tra ping trước, thoát nếu không kết nối được.
    """
    print(f"[INFO] Đang kết nối tới: {stream_url}")

    try:
        urllib.request.urlopen(stream_url, timeout=4)
    except Exception:
        print(f"[LỖI] Không kết nối được tới {stream_url}")
        print("      Kiểm tra:")
        print("      1. App IP Webcam đã bấm 'Start server' chưa?")
        print(f"      2. IP trong app có đúng là {PHONE_IP} không?")
        print("      3. Laptop và điện thoại có cùng WiFi không?")
        sys.exit(1)

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("[LỖI] OpenCV không mở được stream.")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret or frame is None:
        print("[LỖI] Kết nối được nhưng không đọc được frame.")
        sys.exit(1)

    print("[INFO] ✓ Kết nối IP Webcam thành công!")
    print(f"[INFO] ✓ Độ phân giải: {frame.shape[1]}x{frame.shape[0]}")
    return cap


# ══════════════════════════════════════════════
#  PHẦN 5: VẼ BOUNDING BOX + NHÃN YOLO
# ══════════════════════════════════════════════

def draw_detection(frame, coords, class_id: int, confidence: float,
                   vote_count: int) -> None:
    """Vẽ bounding box, nhãn và thanh vote lên frame."""
    x1, y1, x2, y2 = map(int, coords)
    name  = CLASS_NAMES[class_id]
    color = CLASS_COLORS_BGR[class_id]

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Nhãn + confidence + vote count
    label = f"{name}: {confidence*100:.1f}%  [{vote_count}/{WINDOW_SIZE}]"
    fs, th = 0.65, 2
    (tw, txh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    ly = max(y1, txh + 10)
    cv2.rectangle(frame, (x1, ly-txh-bl-4), (x1+tw+4, ly+bl-4), color, cv2.FILLED)
    cv2.putText(frame, label, (x1+2, ly-bl),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)

    # Thanh vote bar ngay dưới box
    fill = int((x2 - x1) * vote_count / WINDOW_SIZE)
    cv2.rectangle(frame, (x1, y2+2), (x2, y2+8),      (60, 60, 60),  cv2.FILLED)
    cv2.rectangle(frame, (x1, y2+2), (x1+fill, y2+8), color,         cv2.FILLED)


# ══════════════════════════════════════════════
#  PHẦN 6: VÒNG LẶP NHẬN DIỆN REAL-TIME
# ══════════════════════════════════════════════

def run_detection(
    model:    YOLO,
    cap:      cv2.VideoCapture,
    analyzer: RipenessAnalyzer
) -> None:
    """
    Vòng lặp chính:
        Frame → YOLO → Temporal smoothing → Vẽ bbox → Phân tích HSV → Badge độ chín
    Nhấn Q hoặc ESC để thoát.
    """
    print("\n[INFO] ══════════════════════════════════════════════════")
    print("[INFO]  BẮT ĐẦU NHẬN DIỆN – Camera điện thoại")
    print("[INFO]  Nhấn 'Q' hoặc 'ESC' để thoát")
    print("[INFO] ══════════════════════════════════════════════════\n")

    vote_history  = deque(maxlen=WINDOW_SIZE)
    conf_history  = deque(maxlen=WINDOW_SIZE)
    coord_history = deque(maxlen=WINDOW_SIZE)

    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0.0
    reconnect_count = 0

    while True:
        ret, frame = cap.read()

        # ── Xử lý mất kết nối ──
        if not ret or frame is None:
            reconnect_count += 1
            if reconnect_count > 10:
                print("[LỖI] Mất kết nối quá lâu, thoát.")
                break
            print(f"[CẢNH BÁO] Mất frame ({reconnect_count}/10), thử lại...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        reconnect_count = 0

        # ── YOLO inference ──
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=0.45,
            verbose=False
        )

        best_cid, best_conf, best_coords = None, 0.0, None
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                if cid < len(CLASS_NAMES) and conf > best_conf:
                    best_conf   = conf
                    best_cid    = cid
                    best_coords = box.xyxy[0].tolist()

        # ── Temporal smoothing: bỏ phiếu theo cửa sổ WINDOW_SIZE frame ──
        vote_history.append(best_cid)
        conf_history.append(best_conf)
        coord_history.append(best_coords)

        valid = [c for c in vote_history if c is not None]

        if len(valid) >= int(WINDOW_SIZE * 0.4):
            counter = Counter(valid)
            top_cid, top_count = counter.most_common(1)[0]

            if top_count / WINDOW_SIZE >= VOTE_RATIO:
                avg_conf = sum(
                    conf_history[i]
                    for i, c in enumerate(vote_history)
                    if c == top_cid
                ) / top_count

                # Lấy tọa độ mới nhất của lớp được vote nhiều nhất
                coords = best_coords if best_cid == top_cid else next(
                    (coord_history[i]
                     for i, c in enumerate(vote_history)
                     if c == top_cid and coord_history[i] is not None),
                    None
                )

                if coords:
                    # ── Vẽ bounding box & nhãn YOLO ──
                    draw_detection(frame, coords, top_cid, avg_conf, top_count)

                    # ── Phân tích độ chín bằng HSV ──
                    class_name = CLASS_NAMES[top_cid]
                    ripeness   = analyzer.analyze(frame, coords, class_name)
                    analyzer.draw_badge(frame, coords, ripeness)

                    # Log ra console
                    print(
                        f"[PHÁT HIỆN] {class_name:<14} "
                        f"conf={avg_conf*100:.1f}%  "
                        f"vote={top_count}/{WINDOW_SIZE}  │  "
                        f"ĐỘ CHÍN: {ripeness.label_vi}  "
                        f"(Hue={ripeness.dominant_hue:.1f}°)"
                    )

        # ── Hiển thị FPS và thông tin kết nối ──
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start   = time.time()

        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"IP Webcam: {PHONE_IP}:{PHONE_PORT}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow("Fruit Detection + Ripeness | Nhan Q de thoat", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            print("\n[INFO] Thoát.")
            break


# ══════════════════════════════════════════════
#  PHẦN 7: ENTRY POINT
# ══════════════════════════════════════════════

if __name__ == "__main__":
    model    = load_model(MODEL_PATH)
    cap      = open_ip_camera(STREAM_URL)
    analyzer = RipenessAnalyzer(roi_shrink=0.15)

    try:
        run_detection(model, cap, analyzer)
    except KeyboardInterrupt:
        print("\n[INFO] Dừng bởi Ctrl+C.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] ✓ Đã giải phóng tài nguyên. Tạm biệt!")