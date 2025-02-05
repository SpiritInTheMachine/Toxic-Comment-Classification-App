import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

# Load dataset (modify if needed)
df = pd.read_csv('CSV/train.csv')

# Preprocessing: Selecting relevant columns
X = df['comment_text']
y = df.iloc[:, 2:]  # Assuming labels are in multiple columns

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF Vectorizer and Model Pipeline
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
model = LogisticRegression(max_iter=1000)

# Convert text into TF-IDF features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Model for Each Toxic Category
models = {}
for category in y.columns:
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_tfidf, y_train[category])
    models[category] = classifier

# Function to classify user input
def classify_text(text):
    processed_text = vectorizer.transform([text])
    predictions = {category: models[category].predict(processed_text)[0] for category in y.columns}
    return predictions

# GUI Application
class ToxicCommentClassifier(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Toxic Comment Classification App")
        self.setGeometry(100, 100, 600, 500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        # Title Label
        self.title_label = QLabel("Toxic Comment Classifier")
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        # Input Field
        self.input_label = QLabel("Enter a comment:")
        self.layout.addWidget(self.input_label)

        self.input_field = QLineEdit()
        self.layout.addWidget(self.input_field)

        # Submit Button
        self.submit_button = QPushButton("Classify")
        self.submit_button.clicked.connect(self.classify_comment)
        self.layout.addWidget(self.submit_button)

        # Output Area
        self.result_label = QLabel("Prediction:")
        self.result_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.result_label)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        self.layout.addWidget(self.result_output)

        # Visualization Buttons
        self.bar_chart_button = QPushButton("Show Toxic Comment Distribution")
        self.bar_chart_button.clicked.connect(self.show_bar_chart)
        self.layout.addWidget(self.bar_chart_button)

        self.pie_chart_button = QPushButton("Show Toxic Category Percentages")
        self.pie_chart_button.clicked.connect(self.show_pie_chart)
        self.layout.addWidget(self.pie_chart_button)

        self.heatmap_button = QPushButton("Show Model Performance Heatmap")
        self.heatmap_button.clicked.connect(self.show_heatmap)
        self.layout.addWidget(self.heatmap_button)

        self.central_widget.setLayout(self.layout)

    def classify_comment(self):
        user_input = self.input_field.text()
        if user_input.strip():
            prediction = classify_text(user_input)
            result_text = "\n".join([f"{key}: {'Toxic' if value == 1 else 'Not Toxic'}" for key, value in prediction.items()])
            self.result_output.setText(f"Prediction:\n{result_text}")
        else:
            self.result_output.setText("Please enter a comment to classify.")

    matplotlib.use("Qt5Agg")

    def show_bar_chart(self):
        category_counts = y.sum().values

        plt.figure(figsize=(8, 5))
        sns.barplot(x=y.columns, y=category_counts, palette="viridis")
        plt.title("Toxic Comment Distribution")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    def show_pie_chart(self):
        category_counts = y.sum().values

        plt.figure(figsize=(6, 6))
        plt.pie(category_counts, labels=y.columns, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
        plt.title("Toxic Category Percentages")
        plt.show()

    def show_heatmap(self):
        y_pred = np.array([models[category].predict(X_test_tfidf) for category in y.columns]).T

        print("Unique values in y_pred:", np.unique(y_pred))  # Debugging check

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0, target_names=y.columns)

        # Ensure correct order of categories
        categories = list(y.columns)
        metrics = ['precision', 'recall', 'f1-score']

        # Extract classification metrics safely
        metrics_matrix = np.array([
            [report[category][metric] if category in report else 0 for metric in metrics]
            for category in categories
        ])

        print("Extracted Metrics Matrix:\n", metrics_matrix)  # Debugging check

        # Generate heatmap
        plt.figure(figsize=(7, 5))
        sns.heatmap(metrics_matrix, annot=True, fmt=".2f", xticklabels=metrics, yticklabels=categories, cmap="coolwarm")
        plt.title("Model Performance Heatmap")
        plt.show()

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ToxicCommentClassifier()
    window.show()
    sys.exit(app.exec_())

