import joblib

# Save the trained model to a file
joblib.dump(clf, 'text_classifier_model.pkl')

# Load the model from the file
clf_loaded = joblib.load('text_classifier_model.pkl')

# Make predictions using the loaded model
y_pred_loaded = clf_loaded.predict(x_test_encoded)

# Print the result
print("Predictions from the loaded model:", y_pred_loaded)
