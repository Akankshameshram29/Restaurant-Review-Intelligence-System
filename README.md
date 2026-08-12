Restaurant Review Intelligence System 🍽️📊
An end-to-end, Aspect-Based Sentiment Analysis (ABSA) Streamlit application designed to extract fine-grained, actionable customer feedback from restaurant reviews.Instead of categorizing an entire review with a single overall label, this system splits complex customer feedback into distinct clauses and identifies sentiment across four core restaurant operational pillars: Food, Service, Ambience, and Price (plus a fallback General category).

🔗 Live Application👉 Access the Deployed App on Streamlit Cloud✨ 

Key Features📝 
1. Single Review ModeAspect-Based Sentiment Extraction:
   Aspect-Based Sentiment Extraction: Parses complex review sentences into clauses and rates each aspect individually.
  
   Confidence Scores: Visualizes confidence percentages alongside sentiment badges (🟢 Positive, 🔴 Negative, 🟡 Neutral).
  
   Overall Sentiment Score: Aggregates individual aspect sentiment ratings into a holistic rating for the review.
  
   Pre-loaded Examples: Includes sample review scenarios for immediate testing.

2. Bulk Dashboard Mode
	  Multi-Format Uploads: Supports batch processing of both CSV and PDF documents.
	  
	  Flexible Column Selection: Allows users to choose which column contains the review text.
	  
	  Real-time Analytics Engine:
	  
	  Key Operational Metrics: Total aspects extracted, positive/negative percentages, and neutral counts.
	  
	  Interactive Plotly Visualizations: Includes an Overall Sentiment Donut Chart and an Aspect-wise Grouped Bar Chart.
	  
	  Complaint Keyword Extraction: Uses custom stopword filtering to isolate recurring operational pain points (e.g., slow, cold, rude).
	  
	  Export Capabilities: Download processed aspect-level structured datasets directly as CSV files for downstream reporting.

🛠️ Tech Stack

	Frontend / Framework: Streamlit
  
	NLP & Data Processing: Python, Pandas, NumPy, spaCy (en_core_web_sm), NLTK, Regular Expressions (re)
  
	Machine Learning: Scikit-Learn (TF-IDF Vectorization, Logistic Regression / Sentiment Classifier)
  
	File Processing: pypdf (PDF extraction)
  
	Visualizations: Plotly Express
  
	Deployment: Streamlit Community Cloud

Project Structure

Restaurant-Review-Intelligence-System/
│
├── data/
│   ├── processed/
│   │   ├── train_processed.csv       # Cleaned and processed training dataset
│   │   └── train_with_sentiment.csv   # Training dataset with engineered sentiment features
│   └── raw/
│       ├── Restaurants_Test_Data.csv # Raw SemEval benchmark test dataset
│       └── Restaurants_Train_v2.csv # Raw SemEval benchmark training dataset
│
├── models/
│   ├── lr_sentiment_model.pkl        # Trained Logistic Regression sentiment classifier
│   └── tfidf_vectorizer.pkl         # Fitted TF-IDF Feature Vectorizer
│
├── notebooks/
│   ├── EDA.ipynb                     # Exploratory Data Analysis & aspect distribution
│   ├── Preprocessing.ipynb           # Text cleaning, lemmatization, and clause splitting
│   ├── model_training.ipynb          # Model training, hyperparameter tuning & evaluation
│   └── sentiment_model.ipynb         # Model persistence and pipeline testing
│
├── app.py                            # Main Streamlit web application & UI logic
├── requirements.txt                  # Python dependencies for production build
├── .gitignore                        # Standard Git exclusion file
└── README.md                         # Project documentation

🚀 Getting Started Locally
Prerequisites
Python 3.9+ installed on your system.
Git installed.

Installation
Clone the repository:
Bash: git clone https://github.com/Akankshameshram29/Restaurant-Review-Intelligence-System.git
cd Restaurant-Review-Intelligence-System

Create and activate a virtual environment:
Windows:
Bash python -m venv venv
venv\Scripts\activate

macOS / Linux:
Bash: python3 -m venv venv
source venv/bin/activate

Install dependencies:
Bash: pip install -r requirements.txt

Launch the Streamlit app:
Bash: streamlit run app.py

Open your browser at http://localhost:8501.

📊 Sample Input & Aspect Breakdown
Given a review like:"The paneer tikka was delicious and the ambience was beautiful, but the waiter took forever and the prices are too high."

The system extracts and categorizes:
	Clause / KeywordTarget AspectExtracted Sentiment
	
	"paneer tikka was delicious" Food🟢 Positive
	
	"ambience was beautiful"Ambience🟢 Positive
	
	"waiter took forever"Service🔴 Negative
	
	"prices are too high"Price🔴 Negative☁️

Deployment 
This app is configured for seamless deployment on Streamlit Community Cloud:

Fork or push changes to GitHub.

Link the repository to share.streamlit.io.

Set app.py as the main entry point.👩‍💻


Author Akanksha Meshram
GitHub: @Akankshameshram29

Live App: restaurant-review-intelligence-system-akanksha.streamlit.app
