import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Sample dataset: list of headlines and labels (positive or negative sentiment)
data = {
    'headline': [
        "Stock prices soar after earnings report",
        "Market crashes amid economic fears",
        "Tech stocks rally throughout the day",
        "Market declines on inflation worries"
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Features and labels
X = df['headline']
y = df['sentiment']

# Convert text data to numerical data
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))

X_vect = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
