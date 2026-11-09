from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score, f1_score

base_dir = Path(__file__).resolve().parent
clean_dir = base_dir / "data" / "clean"
orders = pd.read_csv(clean_dir / "orders.csv")
customers = pd.read_csv(clean_dir / "customers.csv")
products = pd.read_csv(clean_dir / "products.csv")

snapshot = pd.Timestamp("2025-10-01 00:00:00")

completed = orders.loc[orders["status"].eq("completed")].copy()
completed["order_ts"] = pd.to_datetime(completed["order_ts"], errors="coerce")

# Keep rows with valid timestamps for the recency calculation
completed = completed.dropna(subset=["order_ts"])

rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
)

rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
rfm = rfm[["recency", "frequency", "monetary"]].reset_index()

print("Customers:", len(rfm))
print(rfm[["recency", "frequency", "monetary"]].agg(
    ["median", "min", "max", "mean"]
).round(2))


#Cluster with K-means


features = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

results = []
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)

    results.append({
        "K": k,
        "Inertia": model.inertia_,
        "Silhouette": silhouette_score(X_scaled, labels),
    })

scores = pd.DataFrame(results).set_index("K")
print(scores.round(3))


#Profile, name and project the segments 

chosen_k = 4

# Fit final model using X_scaled from the StandardScaler step
final_model = KMeans(
    n_clusters=chosen_k,
    n_init=10,
    random_state=42,
)
rfm = rfm.copy()
rfm["segment_id"] = final_model.fit_predict(X_scaled)

# Segment sizes and mean RFM values
summary = (
    rfm.groupby("segment_id")
    .agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
)

# Set segment names based on each segment's measured profile
premium_id = summary["mean_monetary"].idxmax()
remaining = summary.drop(index=premium_id)

names = {premium_id: "Premium"}
actions = {
    "Premium": "Offer tailored rewards and early access to retain high-value customers."
}

loyal_id = remaining["mean_frequency"].idxmax()
names[loyal_id] = "Loyal Regulars"
actions["Loyal Regulars"] = "Encourage repeat purchases with a loyalty program or subscription offer."
remaining = remaining.drop(index=loyal_id)

atrisk_id = remaining["mean_recency"].idxmax()
names[atrisk_id] = "At-Risk"
actions["At-Risk"] = "Send a re-engagement offer to encourage a return purchase."
remaining = remaining.drop(index=atrisk_id)

growth_id = remaining.index[0]
names[growth_id] = "Growth Potential"
actions["Growth Potential"] = "Use relevant recommendations or bundles to grow order frequency and spend."

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

summary["segment_name"] = summary.index.map(names)
summary["recommended_action"] = summary["segment_name"].map(actions)

print(summary.round(2).to_string())

# PCA on the scaled RFM matrix
pca = PCA(n_components=2)
pca.fit(X_scaled)

print("PCA explained variance ratios:", pca.explained_variance_ratio_.round(3))
print("Total explained variance:", round(pca.explained_variance_ratio_.sum(), 3))

# Save requested columns
output = Path("outputs/customer_segments.csv")
output.parent.mkdir(parents=True, exist_ok=True)

rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
].to_csv(output, index=False)

print(f"Saved: {output}")


#Define the lapse problem

cutoff = pd.Timestamp("2025-09-01")
label_end = pd.Timestamp("2025-10-01")

orders = orders.copy()
customers = customers.copy()

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# Historical data is strictly before the cutoff.
historical = orders.loc[orders["order_ts"] < cutoff].copy()
historical_completed = historical.loc[
    historical["status"].eq("completed")
].copy()

# Cohort: customers with at least one completed order before cutoff.
cohort_ids = historical_completed["customer_id"].unique()

# Label window is used only for y: any completed order in September means not lapsed.
september_completed = orders.loc[
    orders["status"].eq("completed")
    & orders["order_ts"].ge(cutoff)
    & orders["order_ts"].lt(label_end)
]
active_in_september = set(september_completed["customer_id"])
y = pd.Series(
    [int(customer_id not in active_in_september) for customer_id in cohort_ids],
    index=cohort_ids,
    name="lapsed",
)

# Completed-order features from before cutoff.
historical_completed["order_amount"] = pd.to_numeric(
    historical_completed["order_amount"], errors="coerce"
)
historical_completed["discount_pct"] = pd.to_numeric(
    historical_completed["discount_pct"], errors="coerce"
)
historical_completed["delivery_minutes"] = pd.to_numeric(
    historical_completed["delivery_minutes"], errors="coerce"
)

completed_features = (
    historical_completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
    )
)
completed_features["recency"] = (
    cutoff - completed_features["last_order_ts"]
).dt.days
completed_features = completed_features.drop(columns="last_order_ts")

# Most frequent product category among pre-cutoff completed orders.
completed_with_category = historical_completed.drop(
    columns="category",
    errors="ignore",
).merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
)

def most_frequent_category(values):
    modes = values.dropna().mode()
    return modes.iloc[0] if not modes.empty else "Unknown"

category_feature = (
    completed_with_category.groupby("customer_id")["category"]
    .agg(most_frequent_category)
    .rename("most_frequent_category")
)

# Cancellation rate among pre-cutoff orders with a final status.
final_statuses = {"completed", "cancelled"}
historical_final = historical.loc[
    historical["status"].isin(final_statuses)
].copy()
historical_final["is_cancelled"] = historical_final["status"].eq("cancelled")

final_counts = historical_final.groupby("customer_id").agg(
    cancelled_orders=("is_cancelled", "sum"),
    final_orders=("order_id", "nunique"),
)
final_counts["cancellation_rate"] = (
    final_counts["cancelled_orders"] / final_counts["final_orders"]
)

# Customer attributes and tenure at cutoff.
customer_features = customers.set_index("customer_id").copy()
customer_features["tenure_days"] = (
    cutoff - customer_features["signup_date"]
).dt.days

# Assemble rows only for cohort customers. No segment labels are added.
X = (
    customer_features
    .join(completed_features)
    .join(final_counts[["cancellation_rate"]])
    .join(category_feature)
    .loc[cohort_ids]
)

feature_columns = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "most_frequent_category",
    "city",
    "age",
    "gender",
    "tenure_days",
]
X = X[feature_columns].copy()

# Fill missing categories while retaining them as categorical features.
for column in ["most_frequent_category", "city", "gender"]:
    X[column] = X[column].fillna("Unknown")

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# Baseline: always predict "not lapsed" (class 0).
baseline_predictions = [0] * len(y_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
baseline_lapsed_f1 = f1_score(
    y_test, baseline_predictions, pos_label=1, zero_division=0
)

print("Cohort customers:", len(X))
print("Lapsed customers:", int(y.sum()))
print(f"Lapsed rate: {y.mean():.2%}")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print(f"Baseline accuracy: {baseline_accuracy:.3f}")
print(f"Baseline lapsed-class F1: {baseline_lapsed_f1:.3f}")



#Preprocess without leakage

numeric_columns = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]
categorical_columns = ["city", "gender", "most_frequent_category"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

# Fit on training rows only; transform the held-out test rows with that fit.
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Confirm the transformed feature matrices contain no missing or non-finite values.
train_array = (
    X_train_transformed.toarray()
    if hasattr(X_train_transformed, "toarray")
    else X_train_transformed
)
test_array = (
    X_test_transformed.toarray()
    if hasattr(X_test_transformed, "toarray")
    else X_test_transformed
)

assert np.isfinite(train_array).all(), "Training features contain missing or non-finite values"
assert np.isfinite(test_array).all(), "Test features contain missing or non-finite values"

# Also confirm every resulting feature column is complete.
feature_names = preprocessor.get_feature_names_out()
train_features = pd.DataFrame(train_array, columns=feature_names, index=X_train.index)
test_features = pd.DataFrame(test_array, columns=feature_names, index=X_test.index)

assert not train_features.isna().any().any()
assert not test_features.isna().any().any()

# Compare the training-fitted scaler mean with the full-cohort frequency mean.
frequency_index = numeric_columns.index("frequency")
train_frequency_mean = (
    preprocessor.named_transformers_["numeric"]
    .named_steps["scaler"]
    .mean_[frequency_index]
)
cohort_frequency_mean = X["frequency"].mean()

print(f"Training-fitted frequency mean: {train_frequency_mean:.4f}")
print(f"Full-cohort frequency mean:    {cohort_frequency_mean:.4f}")
print("All transformed feature columns are complete.")


#vTrain, tune and evaluate
# Baseline: always predict "not lapsed" (0)
baseline_pred = [0] * len(y_test)
print("Baseline accuracy:", round(accuracy_score(y_test, baseline_pred), 3))
print(
    "Baseline lapsed F1:",
    round(f1_score(y_test, baseline_pred, pos_label=1, zero_division=0), 3),
)

# Logistic regression grid search
logistic_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

logistic_search = GridSearchCV(
    logistic_pipeline,
    param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
logistic_search.fit(X_train, y_train)

# Random forest grid search: 3 depths × 3 leaf sizes × 5 folds = 45 fits
forest_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])

forest_search = GridSearchCV(
    forest_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
forest_search.fit(X_train, y_train)

print("\nLogistic regression")
print("Best parameters:", logistic_search.best_params_)
print("Best CV F1:", round(logistic_search.best_score_, 3))

print("\nRandom forest")
print("Best parameters:", forest_search.best_params_)
print("Best CV F1:", round(forest_search.best_score_, 3))

# Evaluate on the untouched test set only after both searches are complete.
for name, search in [
    ("Logistic regression", logistic_search),
    ("Random forest", forest_search),
]:
    predictions = search.predict(X_test)

    print(f"\n{name} test results")
    print("Confusion matrix (rows=true [0,1], columns=predicted [0,1]):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("Lapsed precision:", round(precision_score(
        y_test, predictions, pos_label=1, zero_division=0
    ), 3))
    print("Lapsed recall:", round(recall_score(
        y_test, predictions, pos_label=1, zero_division=0
    ), 3))
    print("Lapsed F1:", round(f1_score(
        y_test, predictions, pos_label=1, zero_division=0
    ), 3))

    #Class-balance experiment 

    # Preserve the original, unbalanced model and evaluate it on the original test set.
unbalanced_model = logistic_search.best_estimator_
unbalanced_pred = unbalanced_model.predict(X_test)

# Keep every training row, then add replacement samples from the lapsed class.
train_rows = X_train.copy()
train_rows["_lapsed"] = y_train

lapsed_rows = train_rows.loc[train_rows["_lapsed"] == 1]
not_lapsed_count = int((train_rows["_lapsed"] == 0).sum())
lapsed_count = len(lapsed_rows)

duplicates_needed = not_lapsed_count - lapsed_count
added_lapsed_rows = lapsed_rows.sample(
    n=duplicates_needed,
    replace=True,
    random_state=42,
)

balanced_train = pd.concat(
    [train_rows, added_lapsed_rows],
    ignore_index=True,
)

X_train_balanced = balanced_train.drop(columns="_lapsed")
y_train_balanced = balanced_train["_lapsed"]

print("Balanced training class counts:")
print(y_train_balanced.value_counts().sort_index())

# Clone the tuned pipeline so the original fitted model remains available.
balanced_model = clone(unbalanced_model)
balanced_model.fit(X_train_balanced, y_train_balanced)

# Evaluate both models on the same unchanged test set.
balanced_pred = balanced_model.predict(X_test)

def lapsed_metrics(y_true, predictions):
    return {
        "precision": precision_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
    }

comparison = pd.DataFrame({
    "Unbalanced": lapsed_metrics(y_test, unbalanced_pred),
    "Oversampled": lapsed_metrics(y_test, balanced_pred),
})
comparison["Change"] = comparison["Oversampled"] - comparison["Unbalanced"]

print("\nLapsed-class test metrics:")
print(comparison.round(3))

largest_change = comparison["Change"].abs().idxmax()
print(f"\nMetric with the largest absolute change: {largest_change}")


#Interpret, persist and use

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# (a) Feature importance from the tuned random forest
best_rf = forest_search.best_estimator_
rf_preprocessor = best_rf.named_steps["preprocess"]
rf_model = best_rf.named_steps["model"]

rf_feature_names = rf_preprocessor.get_feature_names_out()
rf_importance = (
    pd.Series(rf_model.feature_importances_, index=rf_feature_names)
    .sort_values(ascending=False)
)

print("Top 5 random forest features:")
print(rf_importance.head(5).to_string())

# Coefficient signs from the tuned logistic regression
best_logistic = logistic_search.best_estimator_
log_preprocessor = best_logistic.named_steps["preprocess"]
log_model = best_logistic.named_steps["model"]

log_feature_names = log_preprocessor.get_feature_names_out()
log_coefficients = pd.Series(
    log_model.coef_[0], index=log_feature_names
)

print("\nLogistic regression coefficients:")
for feature in ["recency", "frequency", "monetary"]:
    encoded_name = f"numeric__{feature}"
    coefficient = log_coefficients[encoded_name]
    direction = "positive" if coefficient > 0 else "negative"
    relationship = (
        "higher values are associated with higher predicted lapse probability"
        if coefficient > 0
        else "higher values are associated with lower predicted lapse probability"
    )
    print(
        f"{feature}: {direction} ({coefficient:.4f}) — "
        f"{relationship}, holding other features constant."
    )

# (b) Save the selected model (best random forest), reload, and compare predictions.
selected_model = best_rf
original_test_predictions = selected_model.predict(X_test)

model_path = output_dir / "lapse_model.joblib"
joblib.dump(selected_model, model_path)

reloaded_model = joblib.load(model_path)
reloaded_test_predictions = reloaded_model.predict(X_test)

assert np.array_equal(original_test_predictions, reloaded_test_predictions)
print("\nReloaded model reproduces test predictions exactly:", True)

# (c) Rank Premium customers from the test set by predicted lapse probability.
segments = pd.read_csv(output_dir / "customer_segments.csv")

# Handle is_premium whether CSV parsing reads it as boolean or text.
premium_mask = segments["is_premium"].astype(str).str.lower().isin(
    ["true", "1"]
)
premium_ids = set(segments.loc[premium_mask, "customer_id"])

classes = list(selected_model.classes_)
lapsed_class_index = classes.index(1)
lapse_probabilities = selected_model.predict_proba(X_test)[:, lapsed_class_index]

test_customers = pd.DataFrame({
    "customer_id": X_test.index,
    "lapse_probability": lapse_probabilities,
    "actually_lapsed": y_test.reindex(X_test.index).astype(int).to_numpy(),
})

retention_list = (
    test_customers[test_customers["customer_id"].isin(premium_ids)]
    .sort_values("lapse_probability", ascending=False)
    .head(10)
    [["customer_id", "lapse_probability", "actually_lapsed"]]
)

retention_path = output_dir / "retention_list.csv"
retention_list.to_csv(retention_path, index=False)

print("\nRetention list:")
print(retention_list.to_string(index=False))
print("Truly lapsed in retention list:", int(retention_list["actually_lapsed"].sum()))
print("Saved:", retention_path)