"""
Compare four kinds of models that guess a bean's type from its shape.

Run it with:  python compare_models.py
Everything it makes is saved in the "results" folder.
"""

import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save pictures to files instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

# One fixed "seed" number makes the random parts come out the same every run.
SEED = 42
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# ---------- Step 1: Load the data ----------
dataset = fetch_ucirepo(id=602)
X = dataset.data.features.copy()   # 16 shape measurements
y = dataset.data.targets.squeeze()  # the bean type (7 kinds)
print(f"Loaded {len(X)} beans with {X.shape[1]} measurements each.")


# ---------- Step 2: Split the data ----------
# We hide 20% of the beans as a final "test" set. We do not look at it until
# the very end. "stratify=y" keeps the same mix of bean types in both parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
print(f"Training beans: {len(X_train)}   Test beans: {len(X_test)}")


# ---------- Step 3: Make the models ----------
# Logistic regression needs the numbers scaled (put on the same size range),
# so we put a scaler in front of it. Tree models do not need scaling.
models = {
    "Logistic Regression (baseline)": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
    ),
    "Decision Tree (no limit)": DecisionTreeClassifier(random_state=SEED),
    "Decision Tree (max depth 8)": DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
}


# ---------- Step 4: Test each model fairly ----------
# 5-fold cross-validation: cut the training beans into 5 equal parts. Train on
# 4 parts and check on the 5th. Do that 5 times so each part gets a turn.
# Every model gets the exact same 5 cuts, so the contest is fair.
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

rows = []
for name, model in models.items():
    print(f"\nWorking on: {name}")

    cv = cross_validate(
        model, X_train, y_train, cv=folds,
        scoring=["accuracy", "f1_macro"], return_train_score=True,
    )

    # Now train once on all the training beans and time it.
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    # Time how long it takes to guess all the test beans.
    start = time.perf_counter()
    test_guesses = model.predict(X_test)
    predict_seconds = time.perf_counter() - start

    # How big is the saved model file? Smaller is easier to ship.
    size_kb = len(pickle.dumps(model)) / 1024

    train_acc = cv["train_accuracy"].mean()
    val_acc = cv["test_accuracy"].mean()
    rows.append({
        "Model": name,
        "Train accuracy": train_acc,
        "Validation accuracy (5-fold)": val_acc,
        "Validation spread (+/-)": cv["test_accuracy"].std(),
        "Gap (train - validation)": train_acc - val_acc,
        "Validation F1 (macro)": cv["test_f1_macro"].mean(),
        "Test accuracy": accuracy_score(y_test, test_guesses),
        "Test F1 (macro)": f1_score(y_test, test_guesses, average="macro"),
        "Train time (s)": train_seconds,
        "Predict time per 1,000 beans (ms)": predict_seconds / len(X_test) * 1000 * 1000,
        "Model size (KB)": size_kb,
    })
    print(f"  train {train_acc:.3f} | validation {val_acc:.3f} | "
          f"train time {train_seconds:.2f}s")

    # Save a detailed report for each model, one line per bean type.
    report = classification_report(y_test, test_guesses, digits=3)
    (RESULTS / f"report_{name.split(' (')[0].lower().replace(' ', '_')}"
               f"{'_depth8' if 'depth' in name else ''}.txt").write_text(report)


# ---------- Step 5: Save the comparison table ----------
table = pd.DataFrame(rows)
table.to_csv(RESULTS / "model_comparison.csv", index=False)

pretty = table.copy()
for col in pretty.columns[1:]:
    if col in ("Train time (s)", "Predict time per 1,000 beans (ms)"):
        pretty[col] = pretty[col].map(lambda v: f"{v:.2f}")
    elif col == "Model size (KB)":
        pretty[col] = pretty[col].map(lambda v: f"{v:,.0f}")
    else:
        pretty[col] = pretty[col].map(lambda v: f"{v:.3f}")
(RESULTS / "model_comparison.md").write_text(pretty.to_markdown(index=False))
print("\n" + pretty.to_string(index=False))


# ---------- Step 6: Pictures ----------
# Picture 1: training score next to validation score. A big gap = overfitting.
fig, ax = plt.subplots(figsize=(9, 4.5))
x = range(len(table))
ax.bar([i - 0.2 for i in x], table["Train accuracy"], width=0.4, label="Training")
ax.bar([i + 0.2 for i in x], table["Validation accuracy (5-fold)"], width=0.4,
       label="Validation")
ax.set_xticks(list(x))
ax.set_xticklabels([n.replace(" (", "\n(") for n in table["Model"]], fontsize=8)
ax.set_ylim(0.85, 1.01)
ax.set_ylabel("Accuracy")
ax.set_title("Training vs. validation accuracy (big gap = memorizing)")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "train_vs_validation.png", dpi=150)

# Picture 2: which beans get mixed up. We draw it for the model we recommend
# (logistic regression) and for the top scorer (gradient boosting).
for name, file in [("Logistic Regression (baseline)", "logistic_regression"),
                   ("Gradient Boosting", "gradient_boosting")]:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(models[name], X_test, y_test, ax=ax,
                                          xticks_rotation=45, colorbar=False)
    ax.set_title(f"{name}: right and wrong guesses on test beans")
    fig.tight_layout()
    fig.savefig(RESULTS / f"confusion_matrix_{file}.png", dpi=150)

# Picture 3: which measurements the random forest leaned on most.
forest = models["Random Forest"]
importance = pd.Series(forest.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
importance.plot.barh(ax=ax)
ax.set_title("Which measurements mattered most (Random Forest)")
ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig(RESULTS / "feature_importance.png", dpi=150)
importance.sort_values(ascending=False).to_csv(RESULTS / "feature_importance.csv",
                                               header=["importance"])

print("\nDone. Look in the results folder.")
