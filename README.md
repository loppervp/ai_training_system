# AI Training System for Sales Data Analysis

Một hệ thống AI hoàn chỉnh để phân tích dữ liệu bán hàng, dự đoán xu hướng, và tối ưu hóa kinh doanh.

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Cấu hình](#-cấu-hình)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Mô hình ML](#-mô-hình-machine-learning)
- [Truy vấn AI](#-truy-vấn-ai)
- [Bảo trì & Mở rộng](#-bảo-trì--mở-rộng)
- [Troubleshooting](#-troubleshooting)

## 🚀 Tính năng

### Dữ liệu & ETL
- ✅ Trích xuất dữ liệu từ PostgreSQL (scm_sal_main, scm_sal_data)
- ✅ Biến đổi & làm sạch dữ liệu
- ✅ Kỹ thuật tạo đặc trưng (Feature Engineering)
- ✅ Hỗ trợ batch processing & incremental updates

### Mô hình Machine Learning
- ✅ **Dự đoán thứu nhân (Churn Prediction)**: Xác định khách hàng có nguy cơ mất
- ✅ **Dự báo bán hàng (Sales Forecast)**: Dự báo doanh thu theo ngày/tuần/tháng
- ✅ **Phân khúc khách hàng (Segmentation)**: Phân loại khách hàng theo RFM
- ✅ **Phân tích xu hướng (Trend Analysis)**: Phân tích xu hướng theo sản phẩm, danh mục

### AI & Truy vấn
- ✅ Giao diện truy vấn dựa trên ngôn ngữ tự nhiên (NLP)
- ✅ Tích hợp OpenAI / Anthropic / Google Gemini
- ✅ Hiểu bối cảnh & tham chiếu trước
- ✅ Đơn vị tương tác hoặc batch

### Tự động hóa
- ✅ Lên lịch training hàng ngày/hàng tuần
- ✅ Nhật ký chi tiết (Logging)
- ✅ Xử lý lỗi & khôi phục

## 💻 Yêu cầu hệ thống

### Software
- **Python**: 3.9 trở lên
- **PostgreSQL**: 10 trở lên (hoặc MySQL)
- **Git**: Để quản lý phiên bản

### Hardware
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Disk**: Tối thiểu 5GB cho dữ liệu & mô hình
- **CPU**: 2+ cores

## 📦 Cài đặt

### 1. Clone dự án hoặc tải về

```bash
# Clone từ Git (nếu có)
git clone https://github.com/your-org/ai_training_system.git
cd ai_training_system

# Hoặc tải file zip và giải nén
unzip ai_training_system.zip
cd ai_training_system
```

### 2. Tạo Virtual Environment

```bash
# Với Python venv (khuyến nghị)
python -m venv venv

# Kích hoạt Virtual Environment
# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Cài đặt dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Tạo cấu trúc thư mục

```bash
# Script tự động tạo thư mục
python -c "
import os
dirs = ['data/raw', 'data/processed', 'data/models', 'logs']
for d in dirs:
    os.makedirs(d, exist_ok=True)
print('✓ Tạo thư mục thành công')
"

# Hoặc thủ công
mkdir -p data/raw data/processed data/models logs
```

## ⚙️ Cấu hình

### 1. Cấu hình Database

**File**: `config/database.json`

```json
{
  "source_database": {
    "type": "postgresql",
    "host": "localhost",
    "port": 5432,
    "database": "v57udemo2011",
    "username": "postgres",
    "password": "your_password",
    "schema": "public",
    "pool_size": 10,
    "max_overflow": 20,
    "pool_timeout": 30,
    "echo": false
  },
  "tables": {
    "scm_sal_main": {
      "name": "scm_sal_main",
      "description": "Main sales order table",
      "primary_key": "uniquenum_pri"
    },
    "scm_sal_data": {
      "name": "scm_sal_data",
      "description": "Sales order detail table",
      "primary_key": "uniquenum_uniq",
      "foreign_key": "uniquenum_pri"
    }
  }
}
```

**Thay đổi cần thiết:**
- `host`: Địa chỉ server database
- `port`: Cổng PostgreSQL (mặc định 5432)
- `database`: Tên database
- `username`: Tên user database
- `password`: Mật khẩu database
- `schema`: Schema (mặc định public)

### 2. Cấu hình Environment (.env)

**Tạo file `.env` từ template:**

```bash
cp .env.example .env
```

**Chỉnh sửa `.env`:**

```env
# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=v57udemo2011
DATABASE_USER=postgres
DATABASE_PASSWORD=your_password

# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Settings
LOG_LEVEL=INFO
ENABLE_SCHEDULED_TASKS=True
DEBUG=False
```

**⚠️ QUAN TRỌNG**: Không commit `.env` lên Git!
- `.env` đã được thêm vào `.gitignore`
- Luôn sử dụng `.env.example` như template

### 3. Kiểm tra kết nối Database

```bash
python -c "
from src.extractors.database_extractor import DatabaseExtractor
db = DatabaseExtractor()
if db.test_connection():
    print('✓ Kết nối database thành công')
else:
    print('✗ Lỗi kết nối database')
"
```

## 🎯 Hướng dẫn sử dụng

### Trích xuất dữ liệu

```bash
# Trích xuất toàn bộ dữ liệu
python main.py extract

# Trích xuất theo khoảng thời gian
python main.py extract --date-from 2025-01-01 --date-to 2025-12-31

# Chỉ trích xuất tháng hiện tại
python main.py extract --current-month
```

### Huấn luyện mô hình

```bash
# Huấn luyện tất cả mô hình
python main.py train

# Huấn luyện từng mô hình cụ thể
python main.py train --model churn
python main.py train --model forecast
python main.py train --model segmentation

# Huấn luyện chi tiết
python main.py train --model churn --verbose
```

### Truy vấn AI

```bash
# Chế độ tương tác (Interactive)
python main.py query --interactive

# Truy vấn một lần
python main.py query --query "Top 10 khách hàng"
python main.py query --query "Dự báo doanh thu 30 ngày tới"
python main.py query --query "Sản phẩm bán chạy nhất"

# Với AI provider khác
python main.py query --query "..." --provider anthropic
python main.py query --query "..." --provider google
```

### Lên lịch huấn luyện tự động

```bash
# Chạy scheduled tasks
python main.py scheduled

# Chế độ debug
python main.py scheduled --debug
```

**Lịch mặc định:**
- 02:00 hàng ngày: Huấn luyện incremental
- 03:00 Chủ Nhật: Huấn luyện toàn bộ

### Khởi động Web Server (nếu có)

```bash
python gemini_server.py

# Server sẽ chạy tại http://localhost:5000
```

## 📁 Cấu trúc dự án

```
ai_training_system/
├── .env.example              ← Template biến môi trường
├── .gitignore                ← Git ignore rules
├── requirements.txt          ← Python dependencies
├── README.md                 ← Tài liệu này
├── main.py                   ← Entry point chính
│
├── config/
│   ├── database.json         ← Cấu hình database
│   └── mapping.json          ← JSON mapping cho transform
│
├── src/                      ← Source code
│   ├── extractors/           ← Trích xuất dữ liệu
│   │   ├── database_extractor.py
│   │   └── sales_extractor.py
│   ├── transformers/         ← Biến đổi dữ liệu
│   │   ├── data_transformer.py
│   │   └── feature_engineer.py
│   ├── trainers/             ← Huấn luyện mô hình
│   │   ├── model_trainer.py
│   │   ├── churn_predictor.py
│   │   └── sales_forecaster.py
│   ├── query/                ← Truy vấn AI
│   │   ├── ai_query_interface.py
│   │   ├── intent_parser.py
│   │   └── context_manager.py
│   ├── analysis/             ← Phân tích
│   │   └── product_trend_analyzer.py
│   └── __init__.py
│
├── scripts/
│   └── scheduled_training.py ← Scheduled tasks
│
├── data/                     ← Dữ liệu (git ignored)
│   ├── raw/                  ← Dữ liệu thô từ DB
│   ├── processed/            ← Dữ liệu đã xử lý
│   └── models/               ← Mô hình đã huấn luyện
│
├── logs/                     ← Nhật ký (git ignored)
│   └── scheduled_training.log
│
└── docs/                     ← Tài liệu bổ sung (nếu có)
    └── API.md
```

## 🤖 Mô hình Machine Learning

### 1. Dự đoán Churn (Mất khách hàng)

**Mục đích**: Xác định khách hàng có nguy cơ không mua hàng trong 90 ngày tới

**Đặc trưng sử dụng:**
- `recency`: Số ngày từ lần mua cuối cùng
- `frequency`: Số lần mua trong 90 ngày
- `monetary`: Tổng giá trị mua hàng
- `avg_order_value`: Giá trị đơn hàng trung bình
- Các đặc trưng temporal khác

**Mô hình**: Random Forest Classifier
**Đầu ra**: 
- Xác suất churn (0-1)
- Phân loại rủi ro (Low/High)

**Sử dụng:**
```bash
python main.py train --model churn
```

### 2. Dự báo Bán hàng

**Mục đích**: Dự báo doanh thu cho 7, 14, 30 ngày tới

**Đặc trưng sử dụng:**
- Temporal: năm, tháng, quý, ngày trong tuần
- Cyclical: sin/cos transform của mùa
- Rolling averages: MA-7, MA-30
- Lịch sử doanh thu

**Mô hình**: ARIMA / Prophet / XGBoost
**Đầu ra**: Doanh thu dự báo + khoảng tin cậy

**Sử dụng:**
```bash
python main.py train --model forecast
```

### 3. Phân khúc Khách hàng

**Mục đích**: Phân loại khách hàng thành các nhóm khác nhau

**Phương pháp**: RFM Segmentation
- **R (Recency)**: Mới nhất bao lâu?
- **F (Frequency)**: Bao nhiêu lần mua?
- **M (Monetary)**: Tổng chi tiêu bao nhiêu?

**Phân loại:**
- **Champions** (5,5,5): Mua gần đây, thường xuyên, chi tiêu cao
- **Loyal** (4-5, 4-5, 4-5): Khách hàng trung thành
- **At Risk** (1-2, *, *): Nguy cơ mất
- **Lost** (1, 1, 1): Khách hàng đã mất

## 🧠 Truy vấn AI

### Ví dụ truy vấn

```bash
# Khách hàng
python main.py query --query "Top 10 khách hàng hôm nay"
python main.py query --query "Khách hàng mất nguy cơ cao"
python main.py query --query "Tỷ lệ giữ chân khách hàng"

# Sản phẩm
python main.py query --query "Sản phẩm bán chạy nhất tuần này"
python main.py query --query "Doanh số theo danh mục"
python main.py query --query "Sản phẩm cần nhập kho"

# Xu hướng
python main.py query --query "Xu hướng bán hàng tháng này"
python main.py query --query "Ngày bán chạy nhất trong tuần"
python main.py query --query "So sánh doanh số tháng này vs tháng trước"

# Dự báo
python main.py query --query "Dự báo doanh thu 30 ngày tới"
python main.py query --query "Dựa trên dự báo, cần nhập bao nhiêu hàng?"
```

### Cơ chế hoạt động

1. **Intent Parser**: Phân tích ý định truy vấn
2. **Context Manager**: Lưu trữ bối cảnh cuộc hội thoại
3. **Reference Resolver**: Giải quyết tham chiếu (VD: "hôm nay" → ngày cụ thể)
4. **AI Engine**: Gọi OpenAI/Anthropic/Google
5. **Result Formatter**: Định dạng và trả về kết quả

## 🔧 Bảo trì & Mở rộng

### Thêm bảng dữ liệu mới

1. **Cập nhật `config/database.json`:**

```json
{
  "tables": {
    "inventory_table": {
      "name": "stk_main",
      "description": "Inventory data",
      "primary_key": "id_inv"
    }
  }
}
```

2. **Tạo Extractor mới** trong `src/extractors/`:

```python
class InventoryExtractor(DatabaseExtractor):
    def extract_inventory(self):
        # Logic extraction
        pass
```

3. **Cập nhật `main.py`** để sử dụng extractor mới

4. **Chạy extraction:**
```bash
python main.py extract
```

### Thêm mô hình ML mới

1. **Tạo trainer mới** trong `src/trainers/`:

```python
class CustomModel(BaseTrainer):
    def train(self, data):
        # Training logic
        return model
    
    def predict(self, X):
        # Prediction logic
        return predictions
```

2. **Thêm vào `main.py`:**

```python
if model_type == "custom":
    trainer = CustomModel()
```

3. **Cập nhật scheduled tasks** nếu cần

### Thêm AI provider mới

1. **Chỉnh sửa `src/query/ai_query_interface.py`:**

```python
if provider == "custom":
    response = self.custom_provider.query(prompt)
```

2. **Test:**
```bash
python main.py query --query "..." --provider custom
```

## 📊 Monitoring & Logging

### Xem logs

```bash
# Logs scheduled tasks
tail -f logs/scheduled_training.log

# Logs toàn bộ
tail -f logs/app.log
```

### Cấp độ logging

Trong `.env`:
```env
LOG_LEVEL=DEBUG      # Chi tiết nhất
LOG_LEVEL=INFO       # Thông tin chung
LOG_LEVEL=WARNING    # Cảnh báo
LOG_LEVEL=ERROR      # Lỗi
```

## 🐛 Troubleshooting

### Lỗi kết nối Database

**Lỗi:**
```
psycopg2.OperationalError: could not connect to server
```

**Giải pháp:**
1. Kiểm tra PostgreSQL đang chạy:
   ```bash
   sudo systemctl status postgresql
   ```
2. Kiểm tra cấu hình `.env` và `config/database.json`
3. Kiểm tra firewall/network
4. Test kết nối:
   ```bash
   psql -h localhost -U postgres -d v57udemo2011
   ```

### Lỗi huấn luyện mô hình

**Lỗi:**
```
ValueError: Not enough data for training (minimum 100 records)
```

**Giải pháp:**
1. Kiểm tra dữ liệu extracted:
   ```bash
   ls -lah data/processed/
   ```
2. Trích xuất thêm dữ liệu:
   ```bash
   python main.py extract --date-from 2024-01-01
   ```

### Lỗi API Key

**Lỗi:**
```
AuthenticationError: Invalid API key
```

**Giải pháp:**
1. Kiểm tra `.env`:
   ```bash
   cat .env | grep API_KEY
   ```
2. Đảm bảo API keys hợp lệ
3. Test API key:
   ```python
   python -c "import openai; openai.api_key='...'; print('OK')"
   ```

### Memory không đủ

**Lỗi:**
```
MemoryError: Unable to allocate ...
```

**Giải pháp:**
1. Giảm `BATCH_SIZE` trong `.env`:
   ```env
   BATCH_SIZE=5000  # Giảm từ 10000
   ```
2. Sử dụng chunking:
   ```bash
   python main.py extract --chunk-size 2000
   ```
3. Xóa data cũ:
   ```bash
   rm -rf data/processed/*
   ```

## 📝 Các tệp quan trọng để chú ý

| File | Mục đích | Chỉnh sửa? |
|------|---------|-----------|
| `.env` | Biến môi trường | ✅ Luôn chỉnh sửa |
| `config/database.json` | Cấu hình DB | ✅ Cần chỉnh sửa |
| `requirements.txt` | Dependencies | ❌ Không chỉnh sửa |
| `.gitignore` | Git rules | ❌ Không chỉnh sửa |
| `main.py` | Entry point | ⚠️ Cẩn thận |
| `src/**` | Source code | ⚠️ Cẩn thận |

## 🚀 Best Practices

### 1. Sử dụng Virtual Environment
```bash
source venv/bin/activate  # Luôn luôn
```

### 2. Commit thường xuyên
```bash
git add -A
git commit -m "Thêm feature X"
git push
```

### 3. Backup dữ liệu
```bash
cp -r data/models data/models.backup
```

### 4. Test trước khi deploy
```bash
python main.py extract --date-from 2025-01-01 --date-to 2025-01-07
python main.py train --verbose
python main.py query --query "Test"
```

### 5. Monitoring
- Kiểm tra logs thường xuyên
- Monitor disk space
- Kiểm tra API quotas

## 📞 Hỗ trợ

- **Tài liệu**: Xem file này
- **Logs**: Trong thư mục `logs/`
- **Config**: `config/database.json` và `.env`
- **Issues**: Kiểm tra phần Troubleshooting

## 📄 License

Internal use only. All rights reserved.

---

**Lần cập nhật cuối**: 2025-04-21  
**Phiên bản**: 2.0
