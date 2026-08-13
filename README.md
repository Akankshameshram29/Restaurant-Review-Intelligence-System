# 🍽️ Restaurant Review Intelligence System

An end-to-end NLP system that analyzes restaurant reviews 
and classifies sentiment by aspect — Food, Service, Ambience, 
and Price — using Machine Learning.

🔗 **Live Demo**: https://restaurant-review-intelligence-system-akanksha.streamlit.app/
📁 **GitHub**: https://github.com/Akankshameshram29/Restaurant-Review-Intelligence-System

---

## 📌 Project Overview

Customer reviews contain rich, multi-dimensional feedback — 
a review might praise the food while criticizing the service. 
Traditional sentiment analysis misses this nuance by returning 
a single positive/negative label for the entire review.

This system performs **Aspect-Based Sentiment Analysis (ABSA)** — 
it breaks down each review by aspect and classifies sentiment 
per aspect, giving restaurants and analysts granular, 
actionable insights.

---

## ✨ Features

### 📝 Single Review Mode
- Paste any restaurant review
- Instantly see sentiment per aspect (Food/Service/Ambience/Price)
- Color-coded results with confidence scores
- Overall sentiment summary

### 📊 Bulk Dashboard Mode
- Upload CSV or PDF files containing reviews
- Full analytics dashboard with:
  - Overall sentiment distribution (donut chart)
  - Aspect-wise sentiment breakdown
  - Most praised vs most complained aspects
  - Downloadable sentiment report (CSV)

---

## 🧠 ML Pipeline
Raw Review Text
↓
NLP Preprocessing (spaCy — lemmatization, stopword removal)
↓
Aspect Extraction (rule-based keyword matching)
↓
Sentiment Classification (TF-IDF + Logistic Regression)
↓
Aspect-Sentiment Pairing
↓
Interactive Dashboard (Streamlit)

---

## 📊 Model Performance

| Model | Accuracy | Positive F1 | Negative F1 | Neutral F1 |
|-------|----------|-------------|-------------|------------|
| VADER Baseline | 65.00% | 0.79 | 0.46 | 0.37 |
| **Logistic Regression** | **68.65%** | **0.81** | **0.58** | **0.49** |

Logistic Regression outperforms the VADER baseline by **+3.65%** 
overall, with the biggest improvement on the negative class 
(+12% F1) — the hardest class for lexicon-based approaches.

---

## 🗂️ Dataset

**SemEval 2014 Task 4 — Aspect Based Sentiment Analysis**
- Domain: Restaurant reviews
- Training samples: 3693 aspect-level annotations
- Classes: positive, negative, neutral
- Aspect categories: Food, Service, Ambience, Price, Location

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| NLP Preprocessing | spaCy, NLTK |
| ML Model | scikit-learn (TF-IDF + Logistic Regression) |
| Baseline | VADER SentimentIntensityAnalyzer |
| Frontend | Streamlit |
| Visualization | Plotly |
| PDF Parsing | pypdf |
| Deployment | Streamlit Cloud |

---


## 📊 Model Performance

| Model | Accuracy | Positive F1 | Negative F1 | Neutral F1 |
|-------|----------|-------------|-------------|------------|
| VADER Baseline | 65.00% | 0.79 | 0.46 | 0.37 |
| **Logistic Regression** | **68.65%** | **0.81** | **0.58** | **0.49** |

Logistic Regression outperforms the VADER baseline by **+3.65%** 
overall, with the biggest improvement on the negative class 
(+12% F1) — the hardest class for lexicon-based approaches.

---

## 🗂️ Dataset

**SemEval 2014 Task 4 — Aspect Based Sentiment Analysis**
- Domain: Restaurant reviews
- Training samples: 3693 aspect-level annotations
- Classes: positive, negative, neutral
- Aspect categories: Food, Service, Ambience, Price, Location

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| NLP Preprocessing | spaCy, NLTK |
| ML Model | scikit-learn (TF-IDF + Logistic Regression) |
| Baseline | VADER SentimentIntensityAnalyzer |
| Frontend | Streamlit |
| Visualization | Plotly |
| PDF Parsing | pypdf |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure
Restaurant-Review-Intelligence-System/
│
├── app.py # Streamlit application
├── requirements.txt # Dependencies
│
├── data/
│ ├── raw/ # Original SemEval dataset
│ └── processed/ # Cleaned and mapped data
│
├── models/
│ ├── lr_sentiment_model.pkl # Trained LR model
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer
│
├── notebooks/
│ ├── EDA.ipynb # Exploratory data analysis
│ ├── Preprocessing.ipynb # Text cleaning + aspect mapping
│ ├── sentiment_model.ipynb # VADER baseline
│ └── model_training.ipynb # LR model training
│
└── README.md
---

## 🚀 Run Locally

### Clone Repository
```bash
git clone https://github.com/Akankshameshram29/Restaurant-Review-Intelligence-System.git
cd Restaurant-Review-Intelligence-System
```

### Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run App
```bash
streamlit run app.py
```

---

## 💡 Sample Output

**Input:**
> "Paneer tikka was delicious and the ambience was beautiful, 
> but the waiter took forever and the prices are too high."

**Output:**
| Aspect | Sentiment | Confidence |
|--------|-----------|------------|
| 🟢 Food | Positive | 61.2% |
| 🔴 Service | Negative | 45.9% |
| 🟢 Ambience | Positive | 54.2% |
| 🔴 Price | Negative | 36.1% |

**Overall: 🟡 Mixed**

---

## 🔮 Future Improvements

- Fine-tune BERT for higher accuracy (target: 80%+)
- Add word cloud of common complaint keywords
- Real-time review scraping from Zomato/Google Maps
- Multi-language support (Hindi, Marathi restaurant reviews)
- Confidence threshold filtering for low-confidence predictions

---

## 📚 Learning Outcomes

- Aspect-Based Sentiment Analysis (ABSA)
- NLP preprocessing pipeline (spaCy, NLTK)
- TF-IDF feature extraction
- Supervised ML classification (Logistic Regression)
- VADER lexicon-based sentiment analysis
- Streamlit dashboard development
- Plotly interactive visualizations
- Model evaluation and baseline comparison
- Streamlit Cloud deployment

---

## 👩‍💻 Author

**Akanksha Meshram**
B.Tech — Artificial Intelligence & Data Science

[![GitHub](https://img.shields.io/badge/GitHub-Akankshameshram29-black?logo=github)](https://github.com/Akankshameshram29)

---

## 📄 License

This project is developed for educational and research purposes.

---

