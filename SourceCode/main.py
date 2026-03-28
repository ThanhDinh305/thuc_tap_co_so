"""
============================================================
  Mô tả: Dùng app "IP Webcam" (Android) thay webcam laptop.
         Điện thoại stream video qua WiFi, laptop nhận và
         chạy YOLOv8 nhận diện, hiển thị kết quả trên laptop.
  Cách dùng:
    1. Cài app IP Webcam trên điện thoại Android (CH Play)
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
from ultralytics import YOLO


# ──────────────────────────────────────────────
# BƯỚC 1: CẤU HÌNH – CHỈ CẦN SỬA DÒNG NÀY
# ──────────────────────────────────────────────

# ⚠️ Điền IP điện thoại hiển thị trong app IP Webcam
# VD: "192.168.1.5"  hoặc  "192.168.0.102"
PHONE_IP   = "192.168.2.103"
PHONE_PORT = 8080

# URL stream video từ app IP Webcam (không cần sửa)
STREAM_URL = f"http://{PHONE_IP}:{PHONE_PORT}/video"

MODEL_PATH           = "best.pt"
CONFIDENCE_THRESHOLD = 0.60
WINDOW_SIZE          = 6
VOTE_RATIO           = 0.55

CLASS_NAMES = ['apple', 'avocado', 'banana', 'dragon fruit',
               'lemon', 'mango', 'orange', 'papaya',
               'pineapple', 'strawberry']

CLASS_COLORS_BGR = [
    (0,   51,  255),  # apple        → Đỏ
    (51,  170,  51),  # avocado      → Xanh lá
    (0,   215, 255),  # banana       → Vàng
    (204,  51, 204),  # dragon fruit → Tím
    (68,  255, 255),  # lemon        → Vàng chanh
    (0,   153, 255),  # mango        → Cam
    (0,   102, 255),  # orange       → Cam đậm
    (119, 187, 255),  # papaya       → Cam nhạt
    (136, 255,  68),  # pineapple    → Xanh vàng
    (153,  51, 255),  # strawberry   → Hồng
]


# ──────────────────────────────────────────────
# BƯỚC 2: LOAD MODEL
# ──────────────────────────────────────────────

def load_model(model_path: str) -> YOLO:
    """Load file best.pt với kiểm tra lỗi."""
    if not Path(model_path).exists():
        print(f"[LỖI] Không tìm thấy '{model_path}'")
        sys.exit(1)
    print(f"[INFO] Đang load model ...")
    model = YOLO(model_path)
    print("[INFO] ✓ Load model thành công!")
    return model


# ──────────────────────────────────────────────
# BƯỚC 3: KẾT NỐI IP WEBCAM
# ──────────────────────────────────────────────

def open_ip_camera(stream_url: str) -> cv2.VideoCapture:
    """
    Kết nối tới stream MJPEG từ app IP Webcam.
    Kiểm tra kết nối trước, thoát nếu không ping được.
    """
    print(f"[INFO] Đang kết nối tới: {stream_url}")

    # Kiểm tra app có đang chạy không trước khi mở stream
    try:
        urllib.request.urlopen(stream_url, timeout=4)
    except Exception:
        print(f"[LỖI] Không kết nối được tới {stream_url}")
        print("      Kiểm tra:")
        print("      1. App IP Webcam đã bấm 'Start server' chưa?")
        print(f"      2. IP trong app có đúng là {PHONE_IP} không?")
        print("      3. Laptop và điện thoại có cùng WiFi không?")
        sys.exit(1)

    # Mở stream bằng OpenCV
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("[LỖI] OpenCV không mở được stream.")
        sys.exit(1)

    # Đọc thử 1 frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[LỖI] Kết nối được nhưng không đọc được frame.")
        sys.exit(1)

    print("[INFO] ✓ Kết nối IP Webcam thành công!")
    print(f"[INFO] ✓ Độ phân giải: {frame.shape[1]}x{frame.shape[0]}")
    return cap


# ──────────────────────────────────────────────
# BƯỚC 4: VẼ BOUNDING BOX
# ──────────────────────────────────────────────

def draw_detection(frame, coords, class_id: int, confidence: float,
                   vote_count: int):
    """Vẽ bounding box + nhãn + thanh vote lên frame."""
    x1, y1, x2, y2 = map(int, coords)
    name  = CLASS_NAMES[class_id]
    color = CLASS_COLORS_BGR[class_id]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{name}: {confidence*100:.1f}%  [{vote_count}/{WINDOW_SIZE}]"
    fs, th = 0.65, 2
    (tw, txh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    ly = max(y1, txh + 10)
    cv2.rectangle(frame, (x1, ly-txh-bl-4), (x1+tw+4, ly+bl-4), color, cv2.FILLED)
    cv2.putText(frame, label, (x1+2, ly-bl),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th, cv2.LINE_AA)

    # Thanh vote dưới box
    fill = int((x2-x1) * vote_count / WINDOW_SIZE)
    cv2.rectangle(frame, (x1, y2+2), (x2, y2+8), (60,60,60), cv2.FILLED)
    cv2.rectangle(frame, (x1, y2+2), (x1+fill, y2+8), color, cv2.FILLED)


# ──────────────────────────────────────────────
# BƯỚC 5: VÒNG LẶP NHẬN DIỆN REAL-TIME
# ──────────────────────────────────────────────

def run_detection(model: YOLO, cap: cv2.VideoCapture):
    """
    Vòng lặp chính với Temporal Smoothing.
    Đọc frame từ IP Webcam → YOLO → bỏ phiếu → vẽ → hiển thị.
    """
    print("\n[INFO] ══════════════════════════════════════════")
    print("[INFO]  BẮT ĐẦU NHẬN DIỆN – Camera điện thoại")
    print("[INFO]  Nhấn 'Q' hoặc 'ESC' để thoát")
    print("[INFO] ══════════════════════════════════════════\n")

    vote_history  = deque(maxlen=WINDOW_SIZE)
    conf_history  = deque(maxlen=WINDOW_SIZE)
    coord_history = deque(maxlen=WINDOW_SIZE)

    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0.0
    reconnect_count = 0

    while True:
        ret, frame = cap.read()

        # Nếu mất kết nối → thử kết nối lại tối đa 10 lần
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

        reconnect_count = 0  # Reset nếu đọc frame thành công

        # ── YOLO inference ──
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD,
                                iou=0.45, verbose=False)

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

        # ── Temporal smoothing: đẩy vào hàng đợi bỏ phiếu ──
        vote_history.append(best_cid)
        conf_history.append(best_conf)
        coord_history.append(best_coords)

        valid = [c for c in vote_history if c is not None]
        if len(valid) >= int(WINDOW_SIZE * 0.4):
            counter = Counter(valid)
            top_cid, top_count = counter.most_common(1)[0]

            if top_count / WINDOW_SIZE >= VOTE_RATIO:
                avg_conf = sum(
                    conf_history[i] for i, c in enumerate(vote_history) if c == top_cid
                ) / top_count

                coords = best_coords if best_cid == top_cid else \
                         next((coord_history[i] for i, c in enumerate(vote_history)
                               if c == top_cid and coord_history[i] is not None), None)

                if coords:
                    draw_detection(frame, coords, top_cid, avg_conf, top_count)
                    print(f"[PHÁT HIỆN] {CLASS_NAMES[top_cid]} "
                          f"({avg_conf*100:.1f}%)  vote={top_count}/{WINDOW_SIZE}")

        # ── FPS ──
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start   = time.time()

        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"IP Webcam: {PHONE_IP}:{PHONE_PORT}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2, cv2.LINE_AA)

        cv2.imshow("Fruit Detection - IP Webcam | Nhan Q de thoat", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            print("\n[INFO] Thoát.")
            break


# ──────────────────────────────────────────────
# BƯỚC 6: KHỞI CHẠY
# ──────────────────────────────────────────────

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    cap   = open_ip_camera(STREAM_URL)

    try:
        run_detection(model, cap)
    except KeyboardInterrupt:
        print("\n[INFO] Dừng bởi Ctrl+C.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] ✓ Đã giải phóng tài nguyên. Tạm biệt!")