import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv('data/job_train.csv')

# One-Hot Encoding for 'location' (and any other categorical columns you may have)
df_encoded = pd.get_dummies(df, columns=['location'], drop_first=True)

# Split the data into features (X) and target (y)
X = df_encoded.drop('fraudulent', axis=1)  # Keep text columns for now
y = df_encoded['fraudulent']  # Target column

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill NaN values in text columns with empty strings in both training and test sets
X_train['title'].fillna('', inplace=True)
X_train['description'].fillna('', inplace=True)
X_train['requirements'].fillna('', inplace=True)

X_test['title'].fillna('', inplace=True)
X_test['description'].fillna('', inplace=True)
X_test['requirements'].fillna('', inplace=True)

# Initialize TF-IDF vectorizer for the 'title', 'description', and 'requirements'
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)

# Vectorize the 'title', 'description', and 'requirements' columns separately
X_train_title_tfidf = tfidf.fit_transform(X_train['title']).toarray()
X_test_title_tfidf = tfidf.transform(X_test['title']).toarray()

X_train_description_tfidf = tfidf.fit_transform(X_train['description']).toarray()
X_test_description_tfidf = tfidf.transform(X_test['description']).toarray()

X_train_requirements_tfidf = tfidf.fit_transform(X_train['requirements']).toarray()
X_test_requirements_tfidf = tfidf.transform(X_test['requirements']).toarray()

# Now drop the original text columns from X_train and X_test
X_train_dropped = X_train.drop(['title', 'description', 'requirements'], axis=1).values
X_test_dropped = X_test.drop(['title', 'description', 'requirements'], axis=1).values

# Combine the remaining features with the TF-IDF features (hstack for horizontal stacking)
X_train_combined = np.hstack((X_train_dropped, X_train_title_tfidf, X_train_description_tfidf, X_train_requirements_tfidf))
X_test_combined = np.hstack((X_test_dropped, X_test_title_tfidf, X_test_description_tfidf, X_test_requirements_tfidf))

# Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_combined, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_combined)
print(classification_report(y_test, y_pred))
