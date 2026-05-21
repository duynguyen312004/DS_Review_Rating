# 🍽️ Restaurant Review Sentiment Analysis

Dự án phân loại cảm xúc (Sentiment Analysis) từ các đánh giá nhà hàng bằng phương pháp **TF-IDF + Multinomial Naive Bayes**.

## 🎯 Bài toán

- **Input**: Nội dung bình luận (Review) bằng tiếng Anh
- **Output**: Phân loại Sentiment → `Positive` / `Neutral` / `Negative`
- **Dataset**: [10000 Restaurant Reviews (Kaggle)](https://www.kaggle.com/)

## 📁 Cấu trúc thư mục

```
Project/
├── data/
│   ├── raw/                    # Dữ liệu gốc (không chỉnh sửa)
│   │   └── Restaurant_Reviews.csv
│   └── processed/              # Dữ liệu đã làm sạch
│       └── clean_reviews.csv
├── notebooks/
│   ├── 01_data_overview.ipynb      # Khám phá dataset ban đầu
│   ├── 02_preprocessing.ipynb      # Làm sạch & tạo nhãn sentiment
│   ├── 03_eda.ipynb                # Phân tích khám phá dữ liệu
│   └── 04_train_naive_bayes.ipynb  # Train model & đánh giá
├── src/
│   └── predict.py              # Script dự đoán đơn giản
├── figures/                    # Biểu đồ EDA và evaluation
├── models/                     # Model đã huấn luyện (.pkl)
├── reports/                    # Báo cáo và slide thuyết trình
├── requirements.txt
└── README.md
```

## 🔄 Quy trình thực hiện

| Bước | Notebook | Mô tả |
|------|----------|-------|
| 1 | `01_data_overview.ipynb` | Đọc dataset, kiểm tra cấu trúc, missing values, duplicates |
| 2 | `02_preprocessing.ipynb` | Làm sạch text, tạo cột `sentiment` từ `Rating` |
| 3 | `03_eda.ipynb` | Vẽ biểu đồ phân bố, phân tích dữ liệu |
| 4 | `04_train_naive_bayes.ipynb` | TF-IDF, train Naive Bayes, đánh giá |

## 🏷️ Quy tắc gán nhãn Sentiment

| Rating | Sentiment |
|--------|-----------|
| 1 ⭐, 2 ⭐ | Negative |
| 3 ⭐ | Neutral |
| 4 ⭐, 5 ⭐ | Positive |

## 🚀 Cách chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đặt file `Restaurant_Reviews.csv` vào thư mục `data/raw/`

### 3. Chạy notebook theo thứ tự

```
notebooks/01_data_overview.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_eda.ipynb
notebooks/04_train_naive_bayes.ipynb
```

### 4. Dự đoán review mới

```bash
python src/predict.py "The food was amazing and the staff was very friendly."
```

## 🛠️ Công nghệ sử dụng

- **Python 3.x**
- **Pandas, NumPy** – xử lý dữ liệu
- **Matplotlib, Seaborn** – trực quan hóa
- **Scikit-learn** – TF-IDF, Naive Bayes, đánh giá model
- **Jupyter Notebook** – môi trường làm việc

## 📊 Kết quả (sẽ cập nhật sau khi train)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Multinomial Naive Bayes | - | - |

## 📝 Tác giả

Nhóm 5 – Môn Data Science
