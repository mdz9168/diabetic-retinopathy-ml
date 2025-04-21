from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

def train_logistic_regression(X_train, y_train):
    """
    this trains a logistic regression model.

    Parameters:
    - X_train: training features
    - y_train: training labels

    Returns:
    - Trained LogisticRegression model
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_adaboost_classifier(X_train, y_train):
    """
    this trains an AdaBoost classifier using GridSearchCV for hyperparameter tuning.

    Parameters:
    - X_train: training features
    - y_train: training labels

    Returns:
    - Trained best AdaBoost model
    """
    adaboost = AdaBoostClassifier(random_state=42, algorithm='SAMME')

    param_grid = {
        'n_estimators': [50, 100, 500, 800, 1000, 1151],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2, 1]
    }

    grid_search = GridSearchCV(
        estimator=adaboost,
        param_grid=param_grid,
        cv=4,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("Best AdaBoost Parameters:", grid_search.best_params_)
    return best_model

def train_random_forest_classifier(X_train, y_train, n_estimators=30, random_state=42):
    """
    this trains a Random Forest classifier.

    Parameters:
    - X_train: training features
    - y_train: training labels
    - n_estimators: number of trees in the forest

    Returns:
    - Trained RandomForestClassifier model
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def save_feature_importance(model, feature_names, save_path):
    """
    this saves feature importances to a CSV.

    Parameters:
    - model: trained model with feature_importances_
    - feature_names: list of feature names
    - save_path: file path to save the CSV
    """
    importance = model.feature_importances_
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    df.to_csv(save_path, index=False)
