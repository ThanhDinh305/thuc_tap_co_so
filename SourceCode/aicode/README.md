# 🍎 FruitAI — Hệ Thống Nhận Diện Trái Cây Bằng AI

**FruitAI** là ứng dụng web full-stack sử dụng trí tuệ nhân tạo để nhận diện trái cây qua ảnh tải lên hoặc webcam trực tiếp. Hệ thống phân tích tên loại quả, độ chín và cung cấp thông tin dinh dưỡng chi tiết.

---

## ✨ Tính năng nổi bật

| Tính năng | Mô tả |
|-----------|-------|
| 🤖 **AI nhận diện** | Model YOLOv8 nhận diện 10 loại trái cây với ngưỡng tin cậy 35% |
| 📸 **Webcam trực tiếp** | Chụp ảnh từ camera với giao diện scanner overlay |
| 📤 **Tải ảnh lên** | Kéo thả hoặc chọn file, hỗ trợ JPG/PNG/WebP/BMP, tối đa 10 MB |
| 🌿 **Phân tích độ chín** | Phân tích màu HSV: Chưa chín / Đã chín / Quá chín |
| 🥗 **Thông tin dinh dưỡng** | Calo, protein, chất béo, carbs, chất xơ, vitamin, khoáng chất |
| ⚡ **Cập nhật thời gian thực** | Socket.IO tự động cập nhật lịch sử sau mỗi lần nhận diện |
| 📊 **Dashboard** | Thống kê tổng quan + biểu đồ tuần/tháng + biểu đồ phân phối loại quả |
| 📋 **Lịch sử** | Tìm kiếm, lọc theo ngày, phân trang, xem chi tiết |
| 📥 **Xuất dữ liệu** | Tải file Excel (.xlsx) và PDF |
| 👤 **Hồ sơ** | Cập nhật tên và đổi mật khẩu |
| 🛡️ **Bảng quản trị** | Admin quản lý toàn bộ người dùng và dữ liệu hệ thống |
| 🔒 **Bảo mật** | JWT token, mã hóa mật khẩu bcrypt |

---

## 🍓 Các loại trái cây được hỗ trợ

🍎 Táo &nbsp;|&nbsp; 🥑 Bơ &nbsp;|&nbsp; 🍌 Chuối &nbsp;|&nbsp; 🐉 Thanh Long &nbsp;|&nbsp; 🍋 Chanh &nbsp;|&nbsp; 🥭 Xoài &nbsp;|&nbsp; 🍊 Cam &nbsp;|&nbsp; 🍈 Đu Đủ &nbsp;|&nbsp; 🍍 Dứa &nbsp;|&nbsp; 🍓 Dâu Tây

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                    TRÌNH DUYỆT (port 5173)                  │
│              React + Vite + TailwindCSS                     │
└───────────────────────┬─────────────────────────────────────┘
                        │ REST API + Socket.IO
┌───────────────────────▼─────────────────────────────────────┐
│                  BACKEND (port 5000)                        │
│              Node.js + Express + Socket.IO                  │
└──────────┬──────────────────────────┬───────────────────────┘
           │ HTTP                     │ SQL
┌──────────▼──────────┐   ┌───────────▼──────────────────────┐
│   AI SERVICE        │   │         DATABASE                  │
│   (port 5001)       │   │           MySQL                   │
│  Python Flask       │   │   users + predictions             │
│  YOLOv8 Model       │   └──────────────────────────────────┘
└─────────────────────┘
```

---

## 📁 Cấu trúc thư mục

```
d:\aicode\
├── best.pt                        ← Model YOLOv8 (10 lớp trái cây)
├── nutrition_database.json        ← Cơ sở dữ liệu dinh dưỡng
├── start_all.ps1                  ← Script khởi động tất cả dịch vụ
├── README.md
│
├── ai_service/                    ← Python Flask AI API (port 5001)
│   ├── app.py                     ← Flask server, endpoint /predict
│   ├── model_service.py           ← YOLOv8 + phân tích HSV độ chín
│   └── requirements.txt
│
├── backend/                       ← Node.js + Express + Socket.IO (port 5000)
│   ├── src/
│   │   ├── config/db.js           ← Kết nối MySQL
│   │   ├── middleware/auth.js     ← Xác thực JWT
│   │   ├── controllers/           ← Logic nghiệp vụ
│   │   ├── routes/                ← Định nghĩa API endpoints
│   │   └── server.js              ← Entry point
│   ├── .env                       ← Biến môi trường
│   └── package.json
│
├── frontend/                      ← React + Vite + TailwindCSS (port 5173)
│   ├── src/
│   │   ├── components/            ← UI components (Auth, Dashboard, History…)
│   │   ├── context/               ← AuthContext, SocketContext
│   │   └── services/api.js        ← Gọi API tập trung
│   └── package.json
│
└── database/
    └── schema.sql                 ← Tạo bảng MySQL + tài khoản admin mặc định
```

---

## 🚀 Hướng dẫn cài đặt từ đầu (Windows)

> Hướng dẫn này dành cho máy **chưa cài bất kỳ phần mềm nào**. Thực hiện theo đúng thứ tự.

---

### Bước 1 — Cài đặt Anaconda (Python)

1. Tải Anaconda tại: **https://www.anaconda.com/download**
2. Chạy file `.exe` vừa tải → Next → Next → **Install for All Users** (nếu có quyền admin) hoặc Just Me → Finish
3. Mở **Anaconda Prompt** (tìm trong Start Menu)
4. Tạo môi trường ảo mới tên `tdenv` với Python 3.10:

```bash
conda create -n tdenv python=3.10 -y
conda activate tdenv
```

5. Cài đặt các thư viện AI:

```bash
cd d:\aicode\ai_service
pip install -r requirements.txt
```

> ⏳ Bước này có thể mất 5–15 phút tùy tốc độ mạng vì cần tải PyTorch và YOLOv8.

---

### Bước 2 — Cài đặt Node.js

1. Tải Node.js (LTS) tại: **https://nodejs.org**
2. Chạy file `.msi` vừa tải → Next → **chọn "Automatically install necessary tools"** → Install
3. Mở **Command Prompt** mới, kiểm tra đã cài thành công:

```cmd
node --version
npm --version
```

Phải thấy kết quả dạng `v20.x.x` và `10.x.x`.

4. Cài dependencies cho Backend:

```cmd
cd d:\aicode\backend
npm install
```

5. Cài dependencies cho Frontend:

```cmd
cd d:\aicode\frontend
npm install --force
```

---

### Bước 3 — Cài đặt MySQL

1. Tải **MySQL Community Installer** tại: **https://dev.mysql.com/downloads/installer/**
   - Chọn file `mysql-installer-community-x.x.x.msi` (bản lớn hơn ~450 MB)
2. Chạy installer → chọn **Developer Default** → Execute → Next → Next
3. Trong bước cấu hình:
   - **Authentication Method**: chọn `Use Legacy Authentication Method`
   - **Root Password**: nhập `123456` (phải khớp với file `.env`)
4. Hoàn tất cài đặt

5. Kiểm tra MySQL đang chạy: mở **MySQL Workbench** hoặc mở **Command Prompt** và nhập:

```cmd
mysql -u root -p
```
Nhập mật khẩu `123456` → thấy `mysql>` là thành công.

---

### Bước 4 — Tạo database

Mở **MySQL Workbench** → kết nối với `root` / `123456` → mở tab SQL Editor mới → paste và chạy lệnh sau:

```sql
SOURCE d:\aicode\database\schema.sql;
```

Hoặc dùng command line:

```cmd
mysql -u root -p123456 < d:\aicode\database\schema.sql
```

> Script này sẽ tự tạo database `fruit_recognition_db`, tạo 2 bảng `users` + `predictions` và tạo tài khoản admin mặc định.

---

### Bước 5 — Tạo file cấu hình `.env`

> ⚠️ File `.env` chứa thông tin nhạy cảm nên **không được lưu trong Git**. Khi clone về, bạn cần tự tạo từ file mẫu.

**5.1** Copy file mẫu thành file thật:

```cmd
cd d:\aicode\backend
copy .env.example .env
```

**5.2** Mở file `d:\aicode\backend\.env` bằng Notepad (hoặc VS Code) và điền giá trị thật:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=123456          ← mật khẩu MySQL bạn đặt ở Bước 3
DB_NAME=fruit_recognition_db
JWT_SECRET=thay_bang_chuoi_bat_ky_dai_va_phuc_tap
AI_SERVICE_URL=http://localhost:5001
```

> 💡 `JWT_SECRET` có thể là bất kỳ chuỗi ký tự nào, càng dài càng tốt. Ví dụ: `myfruitai_secret_2024_xyzABC`.

> Nếu bạn đặt mật khẩu MySQL khác `123456`, hãy sửa `DB_PASSWORD` cho khớp.

---

### Bước 6 — Khởi động ứng dụng

#### Cách 1: Chạy tất cả bằng 1 lệnh (PowerShell)

Mở **PowerShell** với quyền Admin → chạy:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
cd d:\aicode
.\start_all.ps1
```

Script sẽ tự mở 3 cửa sổ terminal riêng biệt cho 3 dịch vụ.

#### Cách 2: Chạy thủ công (3 terminal riêng)

**Terminal 1 — AI Service** (dùng Anaconda Prompt):
```bash
conda activate tdenv
cd d:\aicode\ai_service
python app.py
```

**Terminal 2 — Backend** (dùng Command Prompt / PowerShell):
```cmd
cd d:\aicode\backend
npm run dev
```

**Terminal 3 — Frontend** (dùng Command Prompt / PowerShell):
```cmd
cd d:\aicode\frontend
npm run dev
```

---

### Bước 7 — Truy cập ứng dụng

Sau khi cả 3 terminal hiển thị trạng thái **đang chạy**, mở trình duyệt:

| Dịch vụ | Địa chỉ |
|---------|---------|
| 🌐 **Giao diện web** | http://localhost:5173 |
| 🟢 **Backend API** | http://localhost:5000/api/health |
| 🐍 **AI Service** | http://localhost:5001/health |

---

## 🔐 Tài khoản mặc định

| Vai trò | Email | Mật khẩu |
|---------|-------|-----------|
| 👑 Admin | admin@admin.com | admin123 |

> Sau khi đăng nhập bằng tài khoản admin, bạn sẽ thấy mục **Admin Panel** trong thanh điều hướng.

---

## 🌐 API Reference

### Auth
```
POST /api/auth/register       Đăng ký tài khoản mới
POST /api/auth/login          Đăng nhập → trả về JWT
GET  /api/auth/me             Lấy thông tin người dùng (cần auth)
PUT  /api/auth/profile        Cập nhật tên / đổi mật khẩu
```

### Nhận diện
```
POST /api/predictions/upload  Tải ảnh lên (multipart/form-data)
POST /api/predictions/webcam  Ảnh webcam (base64 JSON)
```

### Lịch sử
```
GET    /api/history           Danh sách (có phân trang, tìm kiếm, lọc ngày)
GET    /api/history/:id       Chi tiết 1 bản ghi
DELETE /api/history/:id       Xóa 1 bản ghi
DELETE /api/history           Xóa toàn bộ
GET    /api/history/export/excel   Xuất Excel
GET    /api/history/export/pdf     Xuất PDF
```

### Dashboard
```
GET /api/dashboard/stats        Thống kê tổng quan
GET /api/dashboard/weekly       Dữ liệu biểu đồ 7 ngày
GET /api/dashboard/monthly      Dữ liệu biểu đồ 12 tháng
GET /api/dashboard/distribution Phân phối loại quả
```

### Admin (chỉ tài khoản admin)
```
GET    /api/admin/users         Danh sách người dùng
DELETE /api/admin/users/:id     Xóa người dùng
GET    /api/admin/records       Tất cả bản ghi hệ thống
DELETE /api/admin/records/:id   Xóa bản ghi
GET    /api/admin/stats         Thống kê toàn hệ thống
```

---

## 🛠️ Xử lý lỗi thường gặp

### ❌ AI Service không khởi động được
- Kiểm tra môi trường Anaconda: `conda activate tdenv`
- Kiểm tra file model tồn tại: `d:\aicode\best.pt`
- Thử cài lại: `pip install -r requirements.txt --upgrade`

### ❌ Backend báo lỗi kết nối MySQL
- Kiểm tra MySQL service đang chạy: mở **Services** → tìm `MySQL80` → Start
- Kiểm tra mật khẩu trong `d:\aicode\backend\.env` khớp với MySQL

### ❌ Frontend báo lỗi khi `npm install`
- Thử: `npm install --force` hoặc `npm install --legacy-peer-deps`
- Xóa thư mục `node_modules` rồi cài lại

### ❌ Không nhận diện được trái cây
- Đảm bảo ảnh chứa **1 loại trái cây rõ ràng**, ánh sáng đủ
- File ảnh phải là JPG, PNG, WebP hoặc BMP, kích thước < 10 MB
- AI Service phải đang chạy ở port 5001

---

## 🧰 Công nghệ sử dụng

| Tầng | Công nghệ |
|------|-----------|
| **AI Model** | YOLOv8 (Ultralytics), OpenCV, NumPy |
| **AI Service** | Python 3.10, Flask 3.0, Flask-CORS |
| **Backend** | Node.js 20, Express 4, Socket.IO 4, MySQL2, JWT, Multer, ExcelJS, jsPDF |
| **Frontend** | React 18, Vite 5, TailwindCSS 3, Chart.js, Axios, React-Webcam |
| **Database** | MySQL 8 |
| **OS** | Windows 10/11 |
