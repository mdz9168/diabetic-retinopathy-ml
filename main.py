from src.preprocessing import load_and_preprocess_data
from src.models import (
    train_logistic_regression,
    train_adaboost_classifier,
    train_random_forest_classifier
)
from src.evaluation import evaluate_model, plot_confusion_matrix, save_report

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/Retinopathy_Debrecen.csv")

    # ==============================
    # Logistic Regression
    # ==============================
    log_model = train_logistic_regression(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    cm_log, report_log = evaluate_model(y_test, y_pred_log)

    plot_confusion_matrix(
        cm_log,
        labels=['No DR', 'DR'],
        title='Logistic Regression Confusion Matrix',
        save_path='outputs/figures/logistic_confusion_matrix.png'
    )
    save_report(report_log, 'outputs/reports/logistic_classification_report.json')

    # ==============================
    # AdaBoost Classifier
    # ==============================
    ada_model = train_adaboost_classifier(X_train, y_train)
    y_pred_ada = ada_model.predict(X_test)
    cm_ada, report_ada = evaluate_model(y_test, y_pred_ada)

    plot_confusion_matrix(
        cm_ada,
        labels=['No DR', 'DR'],
        title='AdaBoost Confusion Matrix',
        save_path='outputs/figures/adaboost_confusion_matrix.png'
    )
    save_report(report_ada, 'outputs/reports/adaboost_classification_report.json')

    # ==============================
    # Random Forest Classifier
    # ==============================
    rf_model = train_random_forest_classifier(X_train, y_train, n_estimators=30)
    y_pred_rf = rf_model.predict(X_test)
    cm_rf, report_rf = evaluate_model(y_test, y_pred_rf)

    plot_confusion_matrix(
        cm_rf,
        labels=['No DR', 'DR'],
        title='Random Forest Confusion Matrix',
        save_path='outputs/figures/random_forest_confusion_matrix.png'
    )
    save_report(report_rf, 'outputs/reports/random_forest_classification_report.json')

if __name__ == "__main__":
    main()
