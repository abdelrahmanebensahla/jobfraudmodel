import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras import layers

# Load dataset
df = pd.read_csv('data/job_train.csv')

# One-Hot Encoding for 'location' (and any other categorical columns you may have)
df_encoded = pd.get_dummies(df, columns=['location'], drop_first=True)

# Split the data into features (X) and target (y)
X = df_encoded.drop('fraudulent', axis=1)  # Keep text columns for now
y = df_encoded['fraudulent']  # Target column

# Fill NaN values in text columns with empty strings
for col in ['title', 'description', 'requirements']:
    X[col].fillna('', inplace=True)

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)

# Apply TF-IDF on the text columns ('title', 'description', 'requirements')
X_title_tfidf = tfidf.fit_transform(X['title']).toarray()
X_description_tfidf = tfidf.fit_transform(X['description']).toarray()
X_requirements_tfidf = tfidf.fit_transform(X['requirements']).toarray()

# Drop original text columns
X_dropped = X.drop(['title', 'description', 'requirements'], axis=1).values

# Combine TF-IDF features with the rest of the dataset
X_combined = np.hstack((X_dropped, X_title_tfidf, X_description_tfidf, X_requirements_tfidf))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Standardize the data (optional but often helps with neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple feed-forward neural network using TensorFlow
model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(X_train_scaled.shape[1],)),  # Number of features in the input layer
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # To prevent overfitting
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the model
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Classification report
print(classification_report(y_test, y_pred))
