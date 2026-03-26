"""
main.py
-------
Personal Expense Predictor — BYOP Project
Course: Fundamentals of AI and ML

This script runs the full ML pipeline:
  1. Generate synthetic student expense dataset
  2. Train & compare regression models
  3. Save evaluation charts to outputs/

Usage:
    python main.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display/GUI needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("data",    exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── STEP 1: Generate Data ─────────────────────────────────────────────────────
print("=" * 58)
print("  PERSONAL EXPENSE PREDICTOR — BYOP | AI & ML Course")
print("=" * 58)
print("\n[1/3] Generating student expense dataset...")

np.random.seed(42)
N = 300  # 300 monthly records

months = np.tile(np.arange(1, 13), N // 12 + 1)[:N]

# Monthly allowance from family (hostel student range)
allowance = np.random.normal(8000, 2500, N).clip(3000, 20000)

# Hostel / PG rent — usually fixed or semi-fixed
hostel_rent = np.random.normal(4000, 800, N).clip(1500, 9000)

# Canteen + outside food — highly variable for students
food = allowance * np.random.uniform(0.18, 0.30, N) + np.random.normal(0, 300, N)

# Mobile recharge + internet — this one hits every month without fail
recharge = np.random.normal(350, 100, N).clip(149, 999)

# Auto / bus / cab to go home or around city
transport = np.random.normal(500, 250, N).clip(0, 2000)

# Hangouts, movies, snacks with friends
entertainment = allowance * np.random.uniform(0.05, 0.15, N) + np.random.normal(0, 200, N)

# Stationery, printing, lab fees etc.
academic = np.random.normal(300, 150, N).clip(0, 1500)

# Spike in Oct-Nov (Diwali/semester end shopping) and January (new semester)
seasonal_boost = np.where(
    np.isin(months, [10, 11]), np.random.uniform(500, 2000, N),
    np.where(np.isin(months, [1]), np.random.uniform(300, 1000, N), 0)
)

total_expense = (
    hostel_rent + food + recharge + transport +
    entertainment + academic + seasonal_boost +
    np.random.normal(0, 400, N)
).clip(2000)

df = pd.DataFrame({
    "month":         months,
    "allowance":     allowance.round(2),
    "hostel_rent":   hostel_rent.round(2),
    "food":          food.round(2),
    "recharge":      recharge.round(2),
    "transport":     transport.round(2),
    "entertainment": entertainment.round(2),
    "academic":      academic.round(2),
    "total_expense": total_expense.round(2),
})

df.to_csv("data/student_expenses.csv", index=False)
print(f"    Saved → data/student_expenses.csv  ({len(df)} rows, {df.shape[1]} cols)")
print(f"    Avg monthly expense: ₹{df['total_expense'].mean():,.0f}")
print(f"    Range: ₹{df['total_expense'].min():,.0f} – ₹{df['total_expense'].max():,.0f}")

# ── STEP 2: Train & Evaluate ──────────────────────────────────────────────────
print("\n[2/3] Training models and evaluating performance...")

FEATURES = ["month", "allowance", "hostel_rent", "food",
            "recharge", "transport", "entertainment", "academic"]
TARGET   = "total_expense"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

LINEAR = {"Linear Regression", "Ridge Regression"}

models = {
    "Linear Regression":  LinearRegression(),
    "Ridge Regression":   Ridge(alpha=10),
    "Decision Tree":      DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results     = []
predictions = {}

print(f"\n    {'Model':<25} {'MAE (₹)':>10} {'RMSE (₹)':>10} {'R²':>8}  {'CV R²':>8}")
print("    " + "-" * 67)

for name, model in models.items():
    Xtr = X_train_s if name in LINEAR else X_train
    Xte = X_test_s  if name in LINEAR else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    predictions[name] = y_pred

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, Xtr, y_train, cv=5, scoring="r2").mean()

    results.append({"Model": name, "MAE (₹)": round(mae, 2),
                    "RMSE (₹)": round(rmse, 2),
                    "R² Score": round(r2, 4),
                    "CV R² (mean)": round(cv, 4)})

    print(f"    {name:<25} {mae:>10,.0f} {rmse:>10,.0f} {r2:>8.4f}  {cv:>8.4f}")

results_df = pd.DataFrame(results).sort_values("R² Score", ascending=False)
results_df.to_csv("outputs/model_comparison.csv", index=False)

best = results_df.iloc[0]
print(f"\n    Best model → {best['Model']}  "
      f"(R²={best['R² Score']}, MAE=₹{best['MAE (₹)']:,.0f})")

# ── STEP 3: Plots ─────────────────────────────────────────────────────────────
print("\n[3/3] Generating and saving plots...")

# Plot 1 — Model Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Comparison — Student Expense Predictor",
             fontsize=14, fontweight="bold")
for ax, metric, color in zip(axes,
    ["MAE (₹)", "RMSE (₹)", "R² Score"],
    ["#4C72B0", "#DD8452", "#55A868"]):
    ax.barh(results_df["Model"], results_df[metric], color=color, edgecolor="white")
    ax.set_title(metric, fontweight="bold")
    for i, v in enumerate(results_df[metric]):
        label = f"₹{v:,.0f}" if "₹" in metric else f"{v:.4f}"
        ax.text(v * 1.01, i, label, va="center", fontsize=8)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved → outputs/model_comparison.png")

# Plot 2 — Actual vs Predicted
best_preds = predictions[best["Model"]]
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_preds, alpha=0.6, color="#4C72B0",
            edgecolors="white", s=50)
lims = [min(y_test.min(), best_preds.min()) * 0.95,
        max(y_test.max(), best_preds.max()) * 1.05]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
plt.xlabel("Actual Expense (₹)")
plt.ylabel("Predicted Expense (₹)")
plt.title(f"Actual vs Predicted — {best['Model']}", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved → outputs/actual_vs_predicted.png")

# Plot 3 — Feature Importance
rf = models["Random Forest"]
pd.Series(rf.feature_importances_, index=FEATURES).sort_values().plot(
    kind="barh", color="#4C72B0", edgecolor="white",
    title="Feature Importance — Random Forest", figsize=(7, 5))
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved → outputs/feature_importance.png")

# Plot 4 — Correlation Heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, square=True,
            annot_kws={"size": 8})
plt.title("Feature Correlation Heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved → outputs/correlation_heatmap.png")

# Plot 5 — Avg spending by month (seasonal pattern)
monthly_avg = df.groupby("month")["total_expense"].mean()
plt.figure(figsize=(9, 4))
monthly_avg.plot(kind="bar", color="#4C72B0", edgecolor="white")
plt.xticks(ticks=range(12),
           labels=["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"],
           rotation=45)
plt.title("Average Monthly Expense by Month", fontweight="bold")
plt.ylabel("Avg Total Expense (₹)")
plt.tight_layout()
plt.savefig("outputs/seasonal_pattern.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved → outputs/seasonal_pattern.png")

print("\n" + "=" * 58)
print("  Done! All outputs saved to outputs/")
print("=" * 58)
