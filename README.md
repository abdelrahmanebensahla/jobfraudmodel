# Job Listing Fraud Detection

This project is a machine learning solution designed to predict fraudulent job listings using a feed-forward neural network. The dataset is preprocessed by transforming categorical data, applying TF-IDF to text columns, and standardizing the feature space. The model uses TensorFlow and Keras to classify whether a job listing is real or fraudulent based on the features.

## Features

- **Text Processing**: The `title`, `description`, and `requirements` columns are converted into numerical representations using TF-IDF.
- **One-Hot Encoding**: Categorical features like `location` are encoded.
- **Neural Network**: A feed-forward neural network with two hidden layers is used to classify job listings.
- **Model Performance**: After training, the model provides a classification report to evaluate precision, recall, and accuracy.

## Dependencies

Ensure you have the following Python libraries installed:

```bash
pandas
numpy
scikit-learn
tensorflow
```

You can install them using:

```bash
pip install pandas numpy scikit-learn tensorflow
```

## Dataset

The dataset should be a CSV file (`job_train.csv`) containing job listing data, including text columns like `title`, `description`, and `requirements`. The target variable (`fraudulent`) is a binary column that indicates whether a listing is fraudulent (1) or not (0).

### Example Data Columns:
- **title**: Job title
- **description**: Job description
- **requirements**: Required qualifications
- **location**: Job location
- **fraudulent**: Target column (0 or 1)

## Usage

1. **Load the Dataset**: The dataset is loaded using `pandas`.
2. **Preprocessing**: 
   - Categorical columns are one-hot encoded.
   - Text columns (`title`, `description`, `requirements`) are transformed using TF-IDF.
   - The feature matrix is standardized for better model performance.
3. **Model Architecture**: 
   - A feed-forward neural network with ReLU activation functions and dropout layers for regularization.
   - The final layer uses a sigmoid activation function for binary classification.
4. **Training**: 
   - The model is trained on 80% of the dataset, and 20% is reserved for testing.
5. **Evaluation**: 
   - The model's performance is evaluated using accuracy and a detailed classification report.

## How to Run

1. Ensure you have the dataset file named `job_train.csv` in the working directory.
2. Run the Python script to preprocess the data, train the model, and evaluate it:
   ```bash
   python tfmodel.py
   python forestmodel.py
   ```

## Results

After training, the script will output a classification report that includes metrics such as:

- **Precision**
- **Recall**
- **F1-Score**
- **Accuracy**

These metrics will help you understand the model's performance in predicting fraudulent job listings.

---

This README provides an overview of the project's purpose, setup, and usage, which should be helpful for any future users or collaborators.