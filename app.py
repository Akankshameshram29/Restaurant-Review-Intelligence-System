
import pickle
import io
import re
from collections import Counter
from pypdf import PdfReader
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. MUST BE THE VERY FIRST STREAMLIT CALL
st.set_page_config(
    page_title="Restaurant Review Intelligence System",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Restaurant Review Intelligence System")
st.markdown("Analyze restaurant reviews by aspect — **Food**, **Service**, **Ambience**, **Price**")
st.divider()


# 2. CACHED MODEL AND DATA LOADING
@st.cache_resource
def load_models():
    try:
        with open('models/lr_sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Ensure 'lr_sentiment_model.pkl' and 'tfidf_vectorizer.pkl' exist in the 'models/' folder.")
        st.stop()

model, vectorizer = load_models()

# Defined keywords per aspect
ASPECT_KEYWORDS = {
    'Food': [
        'food', 'pizza', 'pasta', 'paneer', 'dish', 'meal', 'taste', 'chicken',
        'fish', 'meat', 'dessert', 'bread', 'tikka', 'curry', 'rice', 'burger',
        'sandwich', 'sauce', 'salad', 'soup', 'steak', 'seafood', 'cheese', 'egg',
        'coffee', 'wine', 'beer', 'drinks', 'cake'
    ],
    'Service': [
        'waiter', 'waitress', 'staff', 'service', 'server', 'host', 'manager',
        'chef', 'wait', 'slow', 'rude', 'attentive', 'friendly', 'reservation',
        'delivery', 'forever', 'seated', 'helpful', 'crew', 'bartender'
    ],
    'Ambience': [
        'ambience', 'ambiance', 'atmosphere', 'decor', 'music', 'vibe',
        'environment', 'beautiful', 'cozy', 'romantic', 'loud', 'noisy',
        'quiet', 'lighting', 'interior', 'comfortable', 'crowded', 'outdoor',
        'indoor', 'view'
    ],
    'Price': [
        'price', 'prices', 'cost', 'expensive', 'cheap', 'bill', 'value',
        'money', 'overpriced', 'affordable', 'pricey', 'high', 'steep',
        'worth', 'costly', 'tab', 'tips'
    ]
}

STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'was', 'were', 'to', 'in', 'on',
    'at', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'of',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'this', 'that',
    'it', 'its', 'my', 'we', 'our', 'you', 'your', 'they', 'them', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'so', 'too', 'very'
])

def get_clauses(review) -> list[str]:
    """Split text into distinct grammatical clauses safely handling non-string types."""
    # Convert input explicitly to string to handle int, float, or NaN values
    review_str = str(review) if pd.notna(review) else ""
    pattern = r'[.!?;\n]+|,\s*|\b(?:but|however|although|whereas|though)\b'
    clauses = re.split(pattern, review_str.lower())
    return [c.strip() for c in clauses if c and len(c.strip()) > 2]

def predict_sentiment(text: str):
    """Predict sentiment using trained LR model."""
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = round(float(np.max(prob)) * 100, 1)
    return pred, confidence

def analyze_review(review: str) -> list[dict]:
    """Extract aspect sentiment using regex word boundary matching."""
    clauses = get_clauses(review)
    results = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        # Match exact whole words (\bkw\b) to avoid false substring matches like 'tab' in 'table'
        pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
        
        matching_clauses = [c for c in clauses if re.search(pattern, c)]
        
        if matching_clauses:
            combined_clause = " & ".join(matching_clauses)
            sentiment, confidence = predict_sentiment(combined_clause)
            
            results.append({
                'aspect': aspect,
                'sentiment': str(sentiment),
                'confidence': confidence,
                'clause': combined_clause
            })

    return results

def process_bulk_reviews(df: pd.DataFrame, text_column: str):
    """Process an entire DataFrame of reviews and aggregate aspect metrics."""
    rows = []
    
    # Ensure text column is treated as string type
    df[text_column] = df[text_column].astype(str)
    
    for idx, row in df.iterrows():
        review_text = row[text_column]
        if not review_text.strip() or review_text.lower() == 'nan':
            continue
            
        aspect_results = analyze_review(review_text)
        
        if not aspect_results:
            pred_sent, pred_conf = predict_sentiment(review_text)
            rows.append({
                'review_id': idx,
                'original_review': review_text,
                'aspect': 'General',
                'sentiment': pred_sent.lower(),
                'confidence': pred_conf,
                'clause': review_text
            })
        else:
            for res in aspect_results:
                rows.append({
                    'review_id': idx,
                    'original_review': review_text,
                    'aspect': res['aspect'],
                    'sentiment': res['sentiment'],
                    'confidence': res['confidence'],
                    'clause': res['clause']
                })
                
    return pd.DataFrame(rows)



def extract_reviews_from_pdf(uploaded_file) -> list[str]:
    """Extract text from PDF pages and split into non-empty review lines/sentences."""
    reader = PdfReader(uploaded_file)
    extracted_text = ""
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
            
    # Split text by line breaks or periods to isolate individual review sentences
    raw_lines = re.split(r'\n+|\.\s+', extracted_text)
    reviews = [line.strip() for line in raw_lines if len(line.strip()) > 10]
    return reviews


st.sidebar.title("📌 Navigation")
mode = st.sidebar.radio(
    "Select Mode:",
    ["📝 Single Review Mode", "📊 Bulk Dashboard Mode"]
)
st.sidebar.divider()
st.sidebar.info("💡 **Bulk Dashboard Mode** lets you analyze entire customer review CSV files in seconds.")



#MODE 1: SINGLE REVIEW
if mode == "📝 Single Review Mode":
    st.divider()

    examples = [
    "Type your own review below or select an example...",
    "Paneer tikka was delicious and the ambience was beautiful, but the waiter took forever and the prices are too high.",
    "The pasta was bland and overcooked, but the staff was very friendly and helpful.",
    "Amazing food and great atmosphere, service was quick and prices were reasonable.",
    "Worst experience ever, rude staff and the food was cold and tasteless.",
    "Lovely cozy place with great music, the desserts were absolutely fantastic!"
    ]

    selected = st.selectbox("💡 Try an example review:", examples)

    default_text = "" if selected == examples[0] else selected
    review_input = st.text_area(
    "Enter your restaurant review:",
    value=default_text,
    height=120,
    placeholder="e.g. The food was amazing but the service was slow..."
   )

    if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
      if not review_input.strip():
        st.warning("Please enter a review to analyze.")
      else:
        results = analyze_review(review_input)
        if not results:
            st.warning("No recognizable aspects found. Try mentioning food, service, ambience, or price explicitly.")
        else:
            st.divider()
            st.subheader("📊 Aspect-wise Sentiment")
            
            # Metric display
            cols = st.columns(min(len(results), 4))
            for i, result in enumerate(results):
                with cols[i % 4]:
                    sentiment = result['sentiment'].lower()
                    if sentiment == 'positive':
                        color = '🟢'
                        delta = f"+{result['confidence']}% confidence"
                    elif sentiment == 'negative':
                        color = '🔴'
                        delta = f"-{result['confidence']}% confidence"
                    else:
                        color = '🟡'
                        delta = f"~{result['confidence']}% confidence"
                    
                    st.metric(
                        label=f"{color} {result['aspect']}",
                        value=sentiment.capitalize(),
                        delta=delta
                    )
                    st.caption(f"*\"{result['clause']}\"*")
            
            st.divider()
            
            # Overall sentiment
            sentiment_scores = {'positive': 1, 'neutral': 0, 'negative': -1}
            avg = np.mean([sentiment_scores.get(r['sentiment'].lower(), 0) for r in results])
            
            if avg > 0.2:
                overall = '🟢 Positive'
                msg = "Customers are generally happy with this restaurant!"
                st.success(f"**Overall Sentiment: {overall}**\n\n{msg}")
            elif avg < -0.2:
                overall = '🔴 Negative'
                msg = "This restaurant has significant areas to improve."
                st.error(f"**Overall Sentiment: {overall}**\n\n{msg}")
            else:
                overall = '🟡 Mixed'
                msg = "This restaurant has both strengths and weaknesses."
                st.info(f"**Overall Sentiment: {overall}**\n\n{msg}")
            
            with st.expander("🔍 View detailed breakdown"):
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)

#MODE 2: BULK DASHBOARD
else:
    st.title("📈 Bulk Review Analytics Dashboard")
    st.markdown("Upload a **CSV** or **PDF** file containing restaurant reviews to generate a multi-aspect intelligence report.")
    st.divider()

    # Allow both CSV and PDF files
    uploaded_file = st.file_uploader("📂 Upload Reviews File", type=["csv", "pdf"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        df_raw = None

        if file_type == "csv":
            df_raw = pd.read_csv(uploaded_file)
            st.write("### Preview Uploaded CSV Data")
            st.dataframe(df_raw.head(3), use_container_width=True)

            text_col = st.selectbox("Select the column containing review text:", df_raw.columns)
            
        elif file_type == "pdf":
            st.info("📄 Processing PDF Document...")
            pdf_reviews = extract_reviews_from_pdf(uploaded_file)
            
            if not pdf_reviews:
                st.error("No readable text lines longer than 10 characters found in the PDF.")
                st.stop()
                
            df_raw = pd.DataFrame({'Review': pdf_reviews})
            text_col = 'Review'
            
            st.write(f"### Extracted {len(pdf_reviews)} Review Sentences from PDF")
            st.dataframe(df_raw.head(5), use_container_width=True)

        # Run Analytics Engine button
        if df_raw is not None:
            if st.button("🚀 Run Analytics Engine", type="primary", use_container_width=True, key="run_engine"):
                with st.spinner("Analyzing aspects and classifying sentiments..."):
                    results_df = process_bulk_reviews(df_raw, text_col)

                st.success("Analysis Complete!")
                st.divider()

                # --- TOP METRICS ROW ---
                m1, m2, m3, m4 = st.columns(4)
                total_aspects = len(results_df)
                pos_cnt = len(results_df[results_df['sentiment'] == 'positive'])
                neg_cnt = len(results_df[results_df['sentiment'] == 'negative'])
                neu_cnt = len(results_df[results_df['sentiment'] == 'neutral'])

                m1.metric("Total Aspects Extracted", total_aspects)
                m2.metric("Positive Mentions", pos_cnt, f"{round(pos_cnt/total_aspects*100, 1) if total_aspects else 0}%")
                m3.metric("Negative Mentions", neg_cnt, f"-{round(neg_cnt/total_aspects*100, 1) if total_aspects else 0}%", delta_color="inverse")
                m4.metric("Neutral Mentions", neu_cnt)

                st.divider()

                # --- CHARTS ROW 1 ---
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    st.subheader("🍩 Overall Sentiment Distribution")
                    sent_counts = results_df['sentiment'].value_counts().reset_index()
                    sent_counts.columns = ['Sentiment', 'Count']
                    fig_pie = px.pie(
                        sent_counts,
                        values='Count',
                        names='Sentiment',
                        hole=0.4,
                        color='Sentiment',
                        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f1c40f'}
                    )
                    fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_chart2:
                    st.subheader("📊 Aspect-wise Breakdown")
                    aspect_df = results_df.groupby(['aspect', 'sentiment']).size().reset_index(name='count')
                    fig_bar = px.bar(
                        aspect_df,
                        x='aspect',
                        y='count',
                        color='sentiment',
                        barmode='group',
                        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f1c40f'}
                    )
                    fig_bar.update_layout(xaxis_title="Aspect", yaxis_title="Mention Count", margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.divider()

                # --- MOST PRAISED / COMPLAINED ---
                col3, col4 = st.columns(2)

                with col3:
                    st.subheader("🏆 Most Praised Aspects")
                    praised = results_df[results_df['sentiment'] == 'positive']['aspect'].value_counts().reset_index()
                    praised.columns = ['Aspect', 'Count']
                    fig_praised = px.bar(praised, x='Aspect', y='Count', color='Aspect',
                                        color_discrete_sequence=px.colors.sequential.Greens_r)
                    st.plotly_chart(fig_praised, use_container_width=True)

                with col4:
                    st.subheader("⚠️ Most Complained Aspects")
                    complained = results_df[results_df['sentiment'] == 'negative']['aspect'].value_counts().reset_index()
                    complained.columns = ['Aspect', 'Count']
                    fig_complained = px.bar(complained, x='Aspect', y='Count', color='Aspect',
                                           color_discrete_sequence=px.colors.sequential.Reds_r)
                    st.plotly_chart(fig_complained, use_container_width=True)

                st.divider()

                # --- WORD CLOUD SECTION ---
                col_wc1, col_wc2 = st.columns(2)

                with col_wc1:
                    st.subheader("☁️ Common Complaint Keywords")
                    negative_text = ' '.join(
                        results_df[results_df['sentiment'] == 'negative']['clause'].astype(str).tolist()
                    )
                    if negative_text.strip():
                        wc_negative = WordCloud(
                            width=600,
                            height=300,
                            background_color='white',
                            colormap='Reds',
                            max_words=50,
                            stopwords=STOPWORDS,
                            collocations=False
                        ).generate(negative_text)
                        fig_wc1, ax1 = plt.subplots(figsize=(8, 4))
                        ax1.imshow(wc_negative, interpolation='bilinear')
                        ax1.axis('off')
                        plt.tight_layout(pad=0)
                        st.pyplot(fig_wc1)
                        plt.close()
                    else:
                        st.info("No negative reviews found to generate word cloud.")

                with col_wc2:
                    st.subheader("☁️ Common Praise Keywords")
                    positive_text = ' '.join(
                        results_df[results_df['sentiment'] == 'positive']['clause'].astype(str).tolist()
                    )
                    if positive_text.strip():
                        wc_positive = WordCloud(
                            width=600,
                            height=300,
                            background_color='white',
                            colormap='Greens',
                            max_words=50,
                            stopwords=STOPWORDS,
                            collocations=False
                        ).generate(positive_text)
                        fig_wc2, ax2 = plt.subplots(figsize=(8, 4))
                        ax2.imshow(wc_positive, interpolation='bilinear')
                        ax2.axis('off')
                        plt.tight_layout(pad=0)
                        st.pyplot(fig_wc2)
                        plt.close()
                    else:
                        st.info("No positive reviews found to generate word cloud.")

                st.divider()

                # --- EXPORT SECTION ---
                st.subheader("📥 Export Analyzed Dataset")
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Full Sentiment Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="restaurant_sentiment_report.csv",
                    mime="text/csv",
                    type="primary",
                    key="download_csv"
                )

with st.sidebar:
    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Model:** Logistic Regression  
    **Accuracy:** 68.65%  
    **Dataset:** SemEval 2014  
    **Built by:** Akanksha Meshram  
    B.Tech — AI & Data Science
    """)