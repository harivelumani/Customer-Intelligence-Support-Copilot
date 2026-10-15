from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")


def rfm_and_segments(customers: pd.DataFrame, orders: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    completed = orders.loc[orders["status"].eq("completed")].copy()
    completed["order_ts"] = pd.to_datetime(completed["order_ts"])
    rfm = completed.groupby("customer_id", as_index=False).agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
    rfm = rfm.drop(columns="last_order")[["customer_id", "recency", "frequency", "monetary"]]
    rfm[["recency", "frequency", "monetary"]] = rfm[
        ["recency", "frequency", "monetary"]
    ].astype(float)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
    k_metrics = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)
        from sklearn.metrics import silhouette_score

        k_metrics.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(scaled, labels)),
        })
    chosen_k = 4
    kmeans = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    rfm["segment_id"] = kmeans.fit_predict(scaled)
    profiles = rfm.groupby("segment_id").agg(
        size=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    premium_id = int(profiles["mean_monetary"].idxmax())
    names: dict[int, str] = {premium_id: "Premium"}
    left = [int(segment) for segment in profiles.index if int(segment) != premium_id]
    frequent_id = int(profiles.loc[left, "mean_frequency"].idxmax())
    names[frequent_id] = "Frequent"
    left.remove(frequent_id)
    at_risk_id = int(profiles.loc[left, "mean_recency"].idxmax())
    names[at_risk_id] = "At-Risk"
    left.remove(at_risk_id)
    names[left[0]] = "Developing"
    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_id"].eq(premium_id)
    segments = rfm[
        ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    ].copy()
    segment_profiles = (
        segments.groupby(["segment_id", "segment_name", "is_premium"], as_index=False)
        .agg(
            size=("customer_id", "size"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
        )
        .round(2)
        .to_dict(orient="records")
    )
    pca = PCA(n_components=2)
    pca.fit(scaled)
    report = {
        "rfm_count": int(len(rfm)),
        "rfm_summary": {
            "median_recency": float(rfm["recency"].median()),
            "minimum_recency": int(rfm["recency"].min()),
            "maximum_recency": int(rfm["recency"].max()),
            "mean_frequency": float(rfm["frequency"].mean()),
            "maximum_frequency": int(rfm["frequency"].max()),
            "mean_monetary": float(rfm["monetary"].mean()),
            "maximum_monetary": float(rfm["monetary"].max()),
        },
        "k_metrics": k_metrics,
        "chosen_k": chosen_k,
        "segment_profiles": segment_profiles,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "segment_names_by_id": {str(key): value for key, value in names.items()},
    }
    return segments, report


def make_lapse_cohort(customers: pd.DataFrame, orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    before = orders.loc[orders["order_ts"].lt(CUTOFF)].copy()
    completed = before.loc[before["status"].eq("completed")].copy()
    cohort_ids = completed["customer_id"].unique()
    completed = completed.merge(products[["product_id", "category"]], on="product_id", validate="many_to_one")
    labels = orders.loc[
        orders["status"].eq("completed")
        & orders["order_ts"].ge(CUTOFF)
        & orders["order_ts"].lt(LABEL_END)
    ].groupby("customer_id")["order_id"].nunique()

    completed_features = completed.groupby("customer_id").agg(
        last_completed=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_discount=("discount_pct", "mean"),
        average_delivery=("delivery_minutes", "mean"),
    )
    completed_features["recency"] = (CUTOFF - completed_features["last_completed"]).dt.days
    favourite = (
        completed.groupby(["customer_id", "category"]).size().rename("count").reset_index()
        .sort_values(["customer_id", "count", "category"], ascending=[True, False, True])
        .drop_duplicates("customer_id")
        .set_index("customer_id")["category"]
    )
    final_status = before.loc[before["status"].isin(["completed", "cancelled"])]
    cancellation = final_status.assign(is_cancelled=final_status["status"].eq("cancelled")).groupby(
        "customer_id"
    )["is_cancelled"].mean()
    customer_details = customers.set_index("customer_id")[["city", "age", "gender", "signup_date"]].copy()
    customer_details["signup_date"] = pd.to_datetime(customer_details["signup_date"])
    features = completed_features.join(favourite.rename("favourite_category"))
    features = features.join(cancellation.rename("cancellation_rate"))
    features = features.join(customer_details, how="left")
    features["tenure_days"] = (CUTOFF - features["signup_date"]).dt.days
    features["lapsed"] = ~features.index.isin(labels.index)
    features["lapsed"] = features["lapsed"].astype(int)
    features = features.drop(columns=["last_completed", "signup_date"])
    features = features.loc[features.index.isin(cohort_ids)]
    features = features.reset_index()
    features = features.rename(columns={"index": "customer_id"})
    return features


def _metric_summary(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_lapsed": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_lapsed": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_lapsed": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def train_lapse_model(cohort: pd.DataFrame, segments: pd.DataFrame) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    target = cohort["lapsed"].astype(int)
    excluded = {"customer_id", "lapsed"}
    feature_names = [column for column in cohort.columns if column not in excluded]
    numeric = [
        "recency", "frequency", "monetary", "average_discount", "cancellation_rate",
        "average_delivery", "age", "tenure_days",
    ]
    categorical = ["favourite_category", "city", "gender"]
    X = cohort[feature_names].copy()
    y = target
    X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
        X, y, cohort["customer_id"], test_size=0.2, stratify=y, random_state=42
    )
    if X.isna().any().any():
        raise ValueError(f"Missing values in lapse features: {X.columns[X.isna().any()].tolist()}")
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    logistic = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    logistic_search = GridSearchCV(
        logistic, {"classifier__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1", n_jobs=1
    )
    logistic_search.fit(X_train, y_train)
    forest = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    forest_search = GridSearchCV(
        forest,
        {
            "classifier__max_depth": [3, 5, 8],
            "classifier__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
        n_jobs=1,
    )
    forest_search.fit(X_train, y_train)

    y_majority = np.zeros(len(y_test), dtype=int)
    baseline = _metric_summary(y_test, y_majority)
    logistic_model = logistic_search.best_estimator_
    forest_model = forest_search.best_estimator_
    logistic_pred = logistic_model.predict(X_test)
    forest_pred = forest_model.predict(X_test)
    logistic_metrics = _metric_summary(y_test, logistic_pred)
    forest_metrics = _metric_summary(y_test, forest_pred)
    selected_name, selected = (
        ("random_forest", forest_model)
        if forest_metrics["f1_lapsed"] >= logistic_metrics["f1_lapsed"]
        else ("logistic_regression", logistic_model)
    )

    # The random-forest importances and logistic coefficients are reported from their fitted pipelines.
    feature_names_out = forest_model.named_steps["preprocess"].get_feature_names_out()
    forest_importances = forest_model.named_steps["classifier"].feature_importances_
    top_features = sorted(
        zip(feature_names_out.tolist(), forest_importances.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    logistic_names = logistic_model.named_steps["preprocess"].get_feature_names_out()
    logistic_coefs = logistic_model.named_steps["classifier"].coef_[0]
    coefficient_indices = np.argsort(np.abs(logistic_coefs))[::-1][:3]
    top_coefficients = [
        {"feature": str(logistic_names[index]), "coefficient": float(logistic_coefs[index])}
        for index in coefficient_indices
    ]

    class_counts = y_train.value_counts()
    majority_class, minority_class = class_counts.idxmax(), class_counts.idxmin()
    gap = int(class_counts[majority_class] - class_counts[minority_class])
    rng = np.random.default_rng(42)
    minority_positions = np.flatnonzero(y_train.to_numpy() == minority_class)
    extra_positions = rng.choice(minority_positions, size=gap, replace=True)
    X_balanced = pd.concat([X_train, X_train.iloc[extra_positions]], ignore_index=True)
    y_balanced = pd.concat([y_train.reset_index(drop=True), y_train.iloc[extra_positions].reset_index(drop=True)])
    balanced_logistic = logistic_search.best_estimator_
    balanced_logistic.fit(X_balanced, y_balanced)
    balanced_pred = balanced_logistic.predict(X_test)
    balanced_metrics = _metric_summary(y_test, balanced_pred)

    joblib.dump(selected, OUT / "lapse_model.joblib")
    reloaded = joblib.load(OUT / "lapse_model.joblib")
    prediction_match = bool(np.array_equal(selected.predict(X_test), reloaded.predict(X_test)))
    if not prediction_match:
        raise AssertionError("Serialized model predictions differ after reload.")
    probability = selected.predict_proba(X_test)[:, 1]
    test_results = pd.DataFrame({
        "customer_id": train_ids.iloc[0:0].tolist() + test_ids.tolist(),
        "lapse_probability": probability,
        "actually_lapsed": y_test.to_numpy(),
    })
    test_results = test_results.merge(
        segments[["customer_id", "is_premium"]], on="customer_id", how="left"
    )
    retention = test_results.loc[test_results["is_premium"].fillna(False)].sort_values(
        ["lapse_probability", "customer_id"], ascending=[False, True]
    ).head(10)
    retention[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv(
        OUT / "retention_list.csv", index=False
    )

    report = {
        "cohort_rows": int(len(cohort)),
        "lapsed_rows": int(y.sum()),
        "lapsed_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": feature_names,
        "no_missing_features": bool(not X.isna().any().any()),
        "scaler_frequency_mean": float(logistic_model.named_steps["preprocess"].named_transformers_[
            "numeric"
        ].mean_[numeric.index("frequency")]),
        "full_cohort_frequency_mean": float(cohort["frequency"].mean()),
        "majority_baseline": baseline,
        "logistic_regression": {
            "best_params": logistic_search.best_params_,
            "best_cv_f1": float(logistic_search.best_score_),
            "test": logistic_metrics,
            "balanced_training_test": balanced_metrics,
            "balanced_train_class_counts": y_balanced.value_counts().sort_index().to_dict(),
            "top_coefficients": top_coefficients,
        },
        "random_forest": {
            "best_params": forest_search.best_params_,
            "best_cv_f1": float(forest_search.best_score_),
            "test": forest_metrics,
            "top_feature_importances": [
                {"feature": feature, "importance": importance} for feature, importance in top_features
            ],
        },
        "selected_model": selected_name,
        "serialized_predictions_match": prediction_match,
        "retention_list_rows": int(len(retention)),
        "retention_list_actual_lapsed": int(retention["actually_lapsed"].sum()),
        "retention_list": retention[
            ["customer_id", "lapse_probability", "actually_lapsed"]
        ].to_dict(orient="records"),
    }
    (OUT / "part_b_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def run() -> dict:
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    segments, clustering = rfm_and_segments(customers, orders)
    segments.to_csv(OUT / "customer_segments.csv", index=False)
    cohort = make_lapse_cohort(customers, orders, products)
    model_report = train_lapse_model(cohort, segments)
    result = {"clustering": clustering, "lapse_prediction": model_report}
    (OUT / "part_b_full_report.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    run()
