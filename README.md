# Customer Sentiment Analyzer

<!--![Sentiment Analyzer Banner](banner_image_link_here)-->
<div><em>
A web application to analyze customer reviews on Amazon products. This tool leverages a full <strong>NLP pipeline</strong> combined with <strong>ML model building</strong> to understand customer sentiment both automatically and manually. It summarizes sentiments, performs trend analysis, and visualizes sentiment distribution using interactive charts. All manual reviews are stored in a database for historical tracking and further data cleaning.
</em></div><br>

[![View Live App](https://img.shields.io/badge/View_Live_App-Click_Here-blue?style=for-the-badge&logo=streamlit)](https://faisal-khann-customer-sentiment-analyzer-app-ulfzmm.streamlit.app/)
  
---

## 📌Table of Contents
- [Business Problem](#business-problem)
- [Features](#features)
- [NLP-Pipeline & Ml Model notebook](#notebook)
- [How It Works](#how-it-works)
- [Technologies Used](#technology-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preprocessing](#Preprocessing-&-Feature-Extraction (Optimized with Swifter))
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Authors & Contact](#authour-contacts)
- [License](#license)

---

## 📌Business Problem

In the era of e-commerce, customers heavily rely on reviews before making a purchase. Amazon products often receive **hundreds or thousands of reviews**, making it challenging for:  

- **Customers** to quickly understand the overall sentiment of a product.  
- **Sellers** to monitor and analyze customer feedback efficiently.  
- **Businesses** to track sentiment trends over time and improve their products.  

Manual analysis of reviews is time-consuming, error-prone, and inefficient.  

**Our solution: Customer Sentiment Analyzer** solves this problem by:  

- Automatically extracting reviews using the product **ASIN**.  
- Classifying sentiment (positive, negative, neutral) for individual reviews.  
- Summarizing sentiment across multiple reviews.  
- Allowing users to **manually input reviews** and analyze sentiment.  
- Uploading **CSV files** containing reviews for batch analysis.  
- Performing **trend analysis and percentage calculation** for product sentiment.  
- Visualizing sentiment with **bar charts** for quick insights.  
- Storing all manual reviews in a **database** for historical tracking and cleaning.  

This allows both customers and businesses to make **data-driven decisions**, save time, and improve customer satisfaction.

---

## 📌Features

- Extract sentiment from reviews using **ASIN**  
- Manual review sentiment analysis  
- CSV upload for bulk sentiment analysis  
- Sentiment trend analysis  
- Percentage of positive, negative, and neutral reviews  
- Bar chart visualization  
- Database to store historical manual reviews  
- Data cleaning options for stored reviews  

---

## 📌NLP-Pipeline & Ml Model notebook
<div><p><em>Below is the notebook of end-to-end workflow that shows how to implement an NLP pipeline with FastText data and how to build an ML model to predict sentiments </em></p></div>

📁 Notebook: [`workbook.ipynb`](https://github.com/Faisal-khann/Customer_Sentiment_Analyzer/blob/main/notebook/workbook.ipynb)

---

## 📌How It Works

1. **ASIN Analysis**: Enter the Amazon product ASIN → App fetches reviews → Sentiment classification → Summary generated.  
2. **Manual Review Input**: Enter your own review → Sentiment is predicted instantly.  
3. **CSV Upload**: Upload a CSV of reviews → Sentiment is predicted for each row → Trend analysis and visualization.  
4. **Database Storage**: All manual reviews are stored in a database → Can be cleaned and reused for further analysis.  

---

## 📌Technologies Used

- **Programming Language:** Python
- **Web Framework:** Flask / Django
- **NLP & Machine Learning:** NLP Pipeline & ML Model Building (for sentiment analysis)
- **Data Processing & Visualization:** Pandas, numpy, Matplotlib, Seaborn
- **Database:** SQLite (for storing manual reviews)
---

## 📌Project Structure

```text
movie-review-sentiment-analyzer/
├── data/                  # Dataset files (ignored in git due to large size)
│   ├── train.ft.txt
│   └── test.ft.txt
├── models-pkl/                # Saved models & pkl
│   ├── final_model.pkl
│   ├── word2vec.model
│   └── tfidf.pkl
├── app.py                 # Streamlit web app
├── workbook.ipynb         # NLP pipeline + Model training script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```
---
## 📌Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/movie-review-sentiment-analyzer.git
cd movie-review-sentiment-analyzer
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK resources**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

## 📌Preprocessing & Feature Extraction (Optimized with Swifter)

<em>This section covers the steps to clean movie reviews, tokenize them safely using the Word2Vec vocabulary, and convert them into document vectors for modeling.</em>

---

### 1️⃣ Clean and Preprocess Text

Steps applied to each review:

1. Lowercase all text  
2. Remove punctuation and special characters  
3. Tokenize sentences into words  
4. Remove stopwords  
5. Lemmatize words  

```python
import swifter
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation & special chars
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords & lemmatize
    return tokens

# Apply preprocessing with Swifter
print("🔹 Preprocessing train and test data...")
train_df["tokens"] = train_df["text"].swifter.apply(preprocess_text)
test_df["tokens"]  = test_df["text"].swifter.apply(preprocess_text)

```
<em>Tip: Swifter automatically detects if parallel processing is possible, significantly speeding up operations on large datasets.</em>

### 2️⃣ Tokenize Reviews Using Word2Vec Vocabulary

After training the Word2Vec model, filter tokens to keep only words present in the vocabulary.
```python
# Extract vocabulary set from the Word2Vec model
vocab_set = set(w2v_model.wv.index_to_key)

def filter_tokens(tokens, vocab):
    if not isinstance(tokens, list):
        return []
    return [w for w in tokens if w in vocab]

all_tokens = all_tokens.swifter.apply(lambda x: filter_tokens(x, vocab_set))

```
Notes:
- Only words present in the Word2Vec vocabulary are kept.
- Handles non-string entries gracefully by returning an empty list.

### 3️⃣ Convert Tokenized Reviews to weighted Document Vectors

Each review is represented as a fixed-size vector by averaging the embeddings of its tokens.
```python
all_texts = all_tokens.swifter.apply(lambda x: " ".join(x)) # Convert list of tokens → space-separated string

tfidf = TfidfVectorizer(min_df=2)
tfidf.fit(all_texts)  # now it works
word2weight = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

# Weighted document vector
def document_vector_weighted(tokens, w2v_model, word2weight):
    vectors = []
    weights = []
    for word in tokens:
        if word in w2v_model.wv and word in word2weight:
            vectors.append(w2v_model.wv[word])
            weights.append(word2weight[word])
    if vectors:
        return np.average(vectors, axis=0, weights=weights)
    else:
        return np.zeros(w2v_model.vector_size, dtype=np.float32)
```
Notes:
- Returns a zero vector for empty token lists to avoid errors.
- Converts each review into a numerical vector suitable for machine learning models.
- Swifter significantly speeds up vectorization for large datasets.

## 📌Screenshots
<div>
  <img width="1815" height="788" alt="Image" src="https://github.com/user-attachments/assets/093b0da3-dd1e-40c9-9987-a659a06696f3" />

<img width="1806" height="493" alt="Image" src="https://github.com/user-attachments/assets/b5219372-7ebd-417e-9fee-4e7befda00ca" />

<img width="1808" height="404" alt="Image" src="https://github.com/user-attachments/assets/fdde688b-b203-4bec-8523-46844e479d19" />

<img width="1803" height="405" alt="Image" src="https://github.com/user-attachments/assets/0b0b63c5-e19f-4b79-a1cb-e8c6972dd4b0" />

<img width="1812" height="811" alt="Image" src="https://github.com/user-attachments/assets/4f589ecf-4c51-4d72-be4c-8e4a04da4ed3" />

<img width="1809" height="598" alt="Image" src="https://github.com/user-attachments/assets/59c31bef-28c5-4563-9150-143355e2474e" />
</div>

## 📌Author & Contact

**Faisal Khan**  
*Data Analyst | Aspiring Data Scientist*

For any questions, collaboration opportunities, or project-related inquiries, feel free to reach out:

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-orange?style=for-the-badge&logo=google-chrome&logoColor=white)](https://personal-portfolio-alpha-lake.vercel.app/)<br>
[![Email](https://img.shields.io/badge/Email-Faisal%20Khan-blue?style=for-the-badge&logo=gmail&logoColor=white)](mailto:thisside.faisalkhan@example.com)<br>
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Faisal%20Khan-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/faisal-khan23)

---
> Made with ❤️ using Jupyter Notebook, Python, nlp & Machine learning.



