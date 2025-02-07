import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

data = pd.read_csv('spam_email_dataset.csv') 

X = data['text']
y = data['label'].map({'ham': 0, 'spam': 1})  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train) 
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("\n--- Spam Email Detection ---")
while True:
    user_input = input("Enter your email content (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the program. Thank you!")
        break

    input_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(input_tfidf)

    if prediction == 1:
        print("Result: This email is *SPAM*.\n")
    else:
        print("Result: This email is *HAM* (Not Spam).\n")
