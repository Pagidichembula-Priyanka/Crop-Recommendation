# # # import pandas as pd
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.ensemble import RandomForestClassifier
# # # from sklearn.metrics import accuracy_score, classification_report


# # # # 1. Load dataset
# # # df = pd.read_csv("data/Crop_recommendation.csv")

# # # print("First 5 rows:")
# # # print(df.head())

# # # print("\nDataset shape:")
# # # print(df.shape)

# # # print("\nColumns:")
# # # print(df.columns.tolist())


# # # # 2. Split features and target
# # # X = df.drop("label", axis=1)
# # # y = df["label"]

# # # print("\nFeatures sample:")
# # # print(X.head())

# # # print("\nTarget sample:")
# # # print(y.head())


# # # # 3. Train-test split
# # # X_train, X_test, y_train, y_test = train_test_split(
# # #     X, y, test_size=0.2, random_state=42
# # # )

# # # print("\nTraining data size:", X_train.shape)
# # # print("Testing data size:", X_test.shape)


# # # # 4. Create model
# # # model = RandomForestClassifier(random_state=42)

# # # # 5. Train model
# # # model.fit(X_train, y_train)

# # # # 6. Predict on test data
# # # y_pred = model.predict(X_test)

# # # # 7. Evaluate
# # # accuracy = accuracy_score(y_test, y_pred)
# # # print("\nAccuracy:", accuracy)

# # # print("\nClassification Report:")
# # # print(classification_report(y_test, y_pred))


# # # # 8. Test with one custom sample
# # # sample_data = [[90, 42, 43, 20.8, 82.0, 6.5, 202.9]]
# # # prediction = model.predict(sample_data)

# # # print("\nPrediction for sample input:", prediction[0])


# # import pandas as pd
# # import joblib

# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score, classification_report


# # # 1. Load dataset
# # df = pd.read_csv("data/Crop_recommendation.csv")

# # # 2. Split features and target
# # X = df.drop("label", axis=1)
# # y = df["label"]

# # # 3. Train-test split
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.2, random_state=42
# # )

# # # 4. Create model
# # model = RandomForestClassifier(random_state=42)

# # # 5. Train model
# # model.fit(X_train, y_train)

# # # 6. Predict on test data
# # y_pred = model.predict(X_test)

# # # 7. Evaluate model
# # accuracy = accuracy_score(y_test, y_pred)
# # print("Accuracy:", accuracy)
# # print("\nClassification Report:\n")
# # print(classification_report(y_test, y_pred))

# # # 8. Save model
# # joblib.dump(model, "models/model.pkl")
# # print("\nModel saved successfully at models/model.pkl")

# # # 9. Load model again
# # loaded_model = joblib.load("models/model.pkl")
# # print("Model loaded successfully")

# # # 10. Test prediction using loaded model
# # sample_data = [[90, 42, 43, 20.8, 82.0, 6.5, 202.9]]
# # prediction = loaded_model.predict(sample_data)

# # print("\nPrediction from loaded model:", prediction[0])


# import pandas as pd
# import joblib
# import mlflow
# import mlflow.sklearn

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score


# # Start MLflow experiment
# mlflow.set_experiment("crop-recommendation")


# # 1. Load dataset
# df = pd.read_csv("data/Crop_recommendation.csv")

# # 2. Split features and target
# X = df.drop("label", axis=1)
# y = df["label"]

# # 3. Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Start MLflow run
# with mlflow.start_run():

#     # Parameters
#     n_estimators = 50
#     max_depth = None

#     # Log parameters
#     mlflow.log_param("n_estimators", n_estimators)
#     mlflow.log_param("max_depth", max_depth)

#     # 4. Create model
#     model = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         random_state=42
#     )

#     # 5. Train model
#     model.fit(X_train, y_train)

#     # 6. Predict
#     y_pred = model.predict(X_test)

#     # 7. Accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy:", accuracy)

#     # Log metric
#     mlflow.log_metric("accuracy", accuracy)

#     # 8. Save model locally
#     joblib.dump(model, "models/model.pkl")

#     # 9. Log model to MLflow
#     mlflow.sklearn.log_model(model, "model")

#     print("Model logged to MLflow")


import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Force MLflow to use a fixed local folder
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("crop-recommendation")

print("Tracking URI:", mlflow.get_tracking_uri())

# Load dataset
df = pd.read_csv("data/Crop_recommendation.csv")

# Split features and target
X = df.drop("label", axis=1)
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    n_estimators = 50
    max_depth = None

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    mlflow.log_metric("accuracy", accuracy)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    mlflow.sklearn.log_model(sk_model=model, name="model")

    print("Model saved successfully")
    print("Model logged to MLflow successfully")