
import pickle
import re
import numpy as np
import pandas as pd
import streamlit as st

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

def get_clauses(review: str) -> list[str]:
    """Split review into clauses along standard grammatical punctuation & conjunctions."""
    pattern = r'[.!?;\n]+|,\s*|\b(?:but|however|although|whereas|though)\b'
    clauses = re.split(pattern, review.lower())
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


st.subheader("📝 Analyze a Review")

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
                