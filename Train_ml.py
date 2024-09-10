from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize the Logistic Regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_encoded, y_train)
# Predict the labels for the test data
y_pred = clf.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)
# Print the accuracy
print(f'Accuracy: {accuracy:.4f}')