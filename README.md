# Restaurant Review Sentiment Analysis

Du an phan loai cam xuc review nha hang bang pipeline don gian:

- Input: noi dung review tieng Anh
- Output: `Negative`, `Neutral`, hoac `Positive`
- Phuong phap: TF-IDF + Multinomial Naive Bayes

## Cau truc project

```text
Project/
|-- data/
|   |-- raw/
|   |   `-- restaurant_reviews.csv
|   `-- processed/
|       `-- clean_reviews.csv
|-- notebooks/
|   |-- 01_data_overview.ipynb
|   |-- 02_preprocessing.ipynb
|   |-- 03_eda.ipynb
|   `-- 04_train_naive_bayes.ipynb
|-- src/
|   `-- predict.py
|-- figures/
|   |-- rating_distribution.png
|   |-- sentiment_distribution.png
|   |-- review_length_distribution.png
|   `-- confusion_matrix.png
|-- models/
|   |-- naive_bayes_model.joblib
|   `-- tfidf_vectorizer.joblib
|-- reports/
|   `-- classification_report.csv
|-- requirements.txt
`-- README.md
```

## Quy trinh thuc hien

1. `01_data_overview.ipynb`: doc du lieu raw, kiem tra hai cot chinh `Review` va `Rating`.
2. `02_preprocessing.ipynb`: lam sach du lieu, chi giu rating nguyen 1-5, tao `review_clean` va `sentiment`.
3. `03_eda.ipynb`: phan tich phan bo rating, sentiment va do dai review.
4. `04_train_naive_bayes.ipynb`: train TF-IDF + Multinomial Naive Bayes va danh gia model.

## Quy tac tao sentiment

| Rating | Sentiment |
| --- | --- |
| 1, 2 | Negative |
| 3 | Neutral |
| 4, 5 | Positive |

## Ket qua sau preprocessing

Sau khi loc missing values, rating khong hop le va duplicate review:

```text
So dong: 9219
So cot: 4
```

Phan bo sentiment:

```text
Positive    5671
Negative    2385
Neutral     1163
```

## Ket qua model baseline

Model: TF-IDF + Multinomial Naive Bayes

```text
Accuracy: 0.8221
```

Nhan xet:

- Model du doan `Positive` tot nhat.
- Model du doan `Negative` kha on.
- Model du doan `Neutral` kem vi lop nay it du lieu va de bi nham voi hai lop con lai.
- Khi bao cao, khong nen chi nhin accuracy; can xem them F1-score va confusion matrix.

## Cach chay

### 1. Cai dependencies

```bash
pip install -r requirements.txt
```

### 2. Chay notebook theo thu tu

```text
notebooks/01_data_overview.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_eda.ipynb
notebooks/04_train_naive_bayes.ipynb
```

Notebook 4 se tao:

```text
models/naive_bayes_model.joblib
models/tfidf_vectorizer.joblib
reports/classification_report.csv
figures/confusion_matrix.png
```

### 3. Du doan review moi

```bash
python src/predict.py "The food was amazing and the staff were very friendly."
```

Vi du output:

```text
Predicted sentiment: Positive
```

## Cong nghe su dung

- Python
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib
- Jupyter Notebook
