
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# Project folders
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "dataset" / "weatherAUS.csv"
OUTPUT = ROOT / "outputs"
VISUAL = ROOT / "visualizations"

OUTPUT.mkdir(exist_ok=True)
VISUAL.mkdir(exist_ok=True)


def main():

    # 1. Load dataset
    print("Loading Weather Australia dataset...")

    if not DATA_FILE.exists():
        print("ERROR: weatherAUS.csv not found!")
        print("Place it inside the dataset folder.")
        return

    df = pd.read_csv(DATA_FILE)

    print("Dataset shape:", df.shape)
    print("\nFirst five records:")
    print(df.head())

    # 2. Data cleaning
    df.dropna(subset=["RainTomorrow"], inplace=True)

    # Remove data leakage column
    df.drop(columns=["RISK_MM"], errors="ignore", inplace=True)

    # Extract month from date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.month
        df.drop(columns=["Date"], inplace=True)

    # 3. Separate input and output
    y = df["RainTomorrow"].map({"No": 0, "Yes": 1})

    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    X = df.drop(columns=["RainTomorrow"])

    # 4. Identify columns
    numerical = X.select_dtypes(include=["number"]).columns
    categorical = X.select_dtypes(exclude=["number"]).columns

    # 5. Handle missing values
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numerical_pipeline, numerical),
        ("categorical", categorical_pipeline, categorical)
    ])

    # 6. Random Forest algorithm
    random_forest = RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", random_forest)
    ])

    # 7. Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 8. Train model
    print("\nTraining Random Forest model...")

    model.fit(X_train, y_train)

    # 9. Prediction
    predictions = model.predict(X_test)

    # 10. Evaluation
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [accuracy, precision, recall, f1]
    })

    results.to_csv(OUTPUT / "evaluation_results.csv", index=False)

    report = classification_report(
        y_test,
        predictions,
        target_names=["No Rain", "Rain"],
        zero_division=0
    )

    with open(OUTPUT / "classification_report.txt", "w") as file:
        file.write(report)

    # Save actual and predicted values
    prediction_data = X_test.copy()
    prediction_data["Actual"] = y_test.map({0: "No", 1: "Yes"})
    prediction_data["Predicted"] = pd.Series(
        predictions, index=X_test.index
    ).map({0: "No", 1: "Yes"})

    prediction_data.to_csv(
        OUTPUT / "predictions.csv", index=False
    )

    # 11. Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Rain", "Rain"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Rain Prediction Confusion Matrix")
    plt.tight_layout()
    plt.savefig(VISUAL / "confusion_matrix.png", dpi=200)
    plt.close()

    # 12. Evaluation graph
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=results,
        x="Metric",
        y="Score"
    )

    plt.ylim(0, 1)
    plt.title("Model Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(VISUAL / "evaluation_metrics.png", dpi=200)
    plt.close()

    # 13. Feature Importance
    feature_names = (
        model.named_steps["preprocessing"].get_feature_names_out()
    )

    importance = (
        model.named_steps["classifier"].feature_importances_
    )

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(
        by="Importance",
        ascending=False
    )

    feature_df.to_csv(
        OUTPUT / "feature_importance.csv",
        index=False
    )

    top_features = feature_df.head(12).sort_values(
        by="Importance"
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.title("Top 12 Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()

    plt.savefig(
        VISUAL / "feature_importance.png",
        dpi=200
    )
    plt.close()

    # 14. Save trained model
    joblib.dump(
        model,
        OUTPUT / "rain_prediction_model.pkl"
    )

    # 15. Final output
    print("\n===================================")
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("===================================")

    print("\nMODEL EVALUATION:")
    print(results.to_string(index=False))

    print("\nCLASSIFICATION REPORT:")
    print(report)

    print("\nFiles saved in outputs folder:")
    print("- evaluation_results.csv")
    print("- classification_report.txt")
    print("- predictions.csv")
    print("- feature_importance.csv")
    print("- rain_prediction_model.pkl")

    print("\nGraphs saved in visualizations folder:")
    print("- confusion_matrix.png")
    print("- evaluation_metrics.png")
    print("- feature_importance.png")


if __name__ == "__main__":
    main()
