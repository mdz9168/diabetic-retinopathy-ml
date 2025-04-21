from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


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