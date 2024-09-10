from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'solver': ['lbfgs', 'liblinear']
}

# Initialize GridSearchCV with LogisticRegression and the parameter grid
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')

# Fit the model with the training data
grid_search.fit(x_train_encoded, y_train)

# Output the best parameters found by GridSearchCV
print("Best parameters found: ", grid_search.best_params_)

print("Best score found: ", grid_search.best_score_)
