from sklearn.model_selection import cross_val_score

# Get the best estimator from grid search
best_clf = grid_search.best_estimator_

# Perform cross-validation with the best estimator
cv_scores = cross_val_score(best_clf, x_train_encoded, y_train, cv=5)

# Print cross-validation scores
print("Cross-validation scores: ", cv_scores)

