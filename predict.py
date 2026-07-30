import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = joblib.load("./models/fake_news_model.pkl")
vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# preprocess_text()
def preprocess_text(text):

    text = text.lower()

    # remove URL's
    text = re.sub(r'https?://\S+|www\.\S+','',text)

    # remove html tags
    text = re.sub(r'<.*?>','',text)

    # remove punctuation
    text = text.translate(str.maketrans('','',string.punctuation))

    # remove numbers
    text = re.sub(r'\d+','',text)

    # remove extra space
    text = re.sub(r'\s+',' ',text).strip()

    # tokenize
    words = text.split()

    # remove stopwords and lemmatize

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# predict_news()

def predict_news(news):

    cleaned = preprocess_text(news)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    if prediction == 0:
        return "Fake News"
    else:
        return "Real News"