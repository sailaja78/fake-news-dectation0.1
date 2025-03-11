import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re, sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, render_template, request

# Ensure NLTK stopwords are downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text Preprocessing Function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))  # Ensure text is a string
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load Dataset
try:
    df = pd.read_csv("fake_or_real_news.csv")
    if df.empty:
        print("Error: Dataset is empty or incorrectly loaded.")
        sys.exit()
except FileNotFoundError:
    print("Error: fake_or_real_news.csv not found. Please download it.")
    sys.exit()

# Ensure required columns exist
if 'text' not in df.columns or 'label' not in df.columns:
    print("Error: Dataset must contain 'text' and 'label' columns.")
    sys.exit()

# Apply Preprocessing
df['text'] = df['text'].apply(preprocess_text)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train.astype(str))
X_test_vectorized = vectorizer.transform(X_test.astype(str))

# Model Training
model = LogisticRegression(max_iter=500)
model.fit(X_train_vectorized, y_train)

# Predictions
y_pred = model.predict(X_test_vectorized)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save Model and Vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully.")

# Flask App
app = Flask(__name__)

# Load the model and vectorizer
try:
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    print("Error: Model or vectorizer not found. Make sure to run the training script first.")
    sys.exit()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detectNews', methods=['POST'])
def detect_news():
    if request.method == 'POST':
        news_text = request.form['news_text']

        # Preprocess the input text
        preprocessed_text = preprocess_text(news_text)

        # Vectorize the preprocessed text
        vectorized_text = vectorizer.transform([preprocessed_text])

        # Make a prediction
        prediction = model.predict(vectorized_text)[0]

        # Render the result
        return render_template('result.html', news_text=news_text, prediction=prediction)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)