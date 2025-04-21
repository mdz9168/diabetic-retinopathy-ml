from src.preprocessing import load_and_preprocess_data
from src.models import train_logistic_regression, train_adaboost_classifier
from src.evaluation import evaluate_model, plot_confusion_matrix, save_report

import pandas as pd

def main():
    # this will load and split data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/Retinopathy_Debrecen.csv")

    # ---- Logistic Regression ----
    log_model = train_logistic_regression(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    cm_log, report_log = evaluate_model(y_test, y_pred_log)
    plot_confusion_matrix(
        cm_log,
        ['No DR', 'DR'],
        'Logistic Regression Confusion Matrix',
        'outputs/figures/logistic_confusion_matrix.png'
    )
    save_report(report_log, 'outputs/reports/logistic_classification_report.json')

    # ---- AdaBoost Classifier ----
    ada_model = train_adaboost_classifier(X_train, y_train)
    y_pred_ada = ada_model.predict(X_test)
    cm_ada, report_ada = evaluate_model(y_test, y_pred_ada)
    plot_confusion_matrix(
        cm_ada,
        ['No DR', 'DR'],
        'AdaBoost Confusion Matrix',
        'outputs/figures/adaboost_confusion_matrix.png'
    )
    save_report(report_ada, 'outputs/reports/adaboost_classification_report.json')

    # ---- AdaBoost Feature Importances ----
    feature_names = (
        X_train.columns
        if hasattr(X_train, "columns")
        else [f"feature_{i}" for i in range(X_train.shape[1])]
    )
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": ada_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    importance_df.to_csv("outputs/reports/adaboost_feature_importance.csv", index=False)
    print("\nTop AdaBoost Feature Importances:\n", importance_df.head())

if __name__ == "__main__":
    main()
