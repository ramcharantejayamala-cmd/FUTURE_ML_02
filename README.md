Future Interns - Machine Learning Task 2

Ticket Classification using Machine Learning

Project Overview

This project builds a Machine Learning model that automatically classifies customer support tickets into their respective categories based on the ticket description. The solution uses Natural Language Processing (NLP) techniques with TF-IDF Vectorization and Logistic Regression for text classification.

⸻

Project Structure

FUTURE_ML_02/
│
├── data/
│   └── tickets.csv
│
├── outputs/
│   ├── accuracy.png
│   └── ticket_classifier.pkl
│
├── screenshots/
│
├── main.py
├── requirements.txt
└── README.md

⸻

Technologies Used

* Python 3
* Pandas
* Scikit-learn
* Matplotlib
* Joblib

⸻

Machine Learning Workflow

1. Load the dataset
2. Clean missing values
3. Split the dataset into training and testing sets
4. Convert text into numerical features using TF-IDF
5. Train a Logistic Regression classifier
6. Evaluate model performance
7. Save the trained model
8. Generate an accuracy graph

⸻

Installation

Install the required libraries:

python -m pip install -r requirements.txt

⸻

Run the Project

python main.py

⸻

Output

The project generates:

* Trained Machine Learning model (ticket_classifier.pkl)
* Accuracy graph (accuracy.png)
* Classification report
* Accuracy score

⸻

Sample Output

Accuracy: 0.91
Classification Report
precision    recall    f1-score
...

⸻

Skills Demonstrated

* Data Preprocessing
* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Logistic Regression
* Model Evaluation
* Data Visualization
* Python Programming

⸻

Author

Name: Yamala Ram charan teja

Internship: Future Interns – Machine Learning Internship

Task: Task 2 – Ticket Classification using Machine Learning