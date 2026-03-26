# 💸 Personal Expense Predictor
### *Because running out of money on the 20th of every month gets old really fast.*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

</div>

---

## 🙋 The Story Behind This Project

Okay so here is the thing. I live in a hostel. Every month my family sends me an allowance and every single month — without fail — I somehow end up broke before the month even ends. 😅

It is not like I am buying anything crazy. It is the usual stuff — canteen food, mobile recharge, auto rides, the occasional outing with friends. But somehow it all adds up and I never see it coming until my UPI shows me a number I do not want to look at.

So I started thinking — what if I could predict this? What if a machine learning model could look at my spending patterns and just tell me upfront: *"hey, this month is going to be expensive, watch out."* 🔮

That thought turned into this project. And honestly, building it taught me more about regression and ML evaluation than any lecture alone could have. This is my BYOP submission for the Fundamentals of AI and ML course — a problem I genuinely face, solved using tools I learned in class.

---

## 📌 What This Project Does

**Personal Expense Predictor** is a machine learning system that predicts a hostel student's total monthly expenses based on:

- 💰 Their monthly allowance from family
- 🏠 Their hostel or PG rent
- 🍱 Food and canteen spending
- 📱 Mobile recharge and internet bills
- 🚌 Transport costs — auto, bus, cab
- 🎬 Entertainment and outings with friends
- 📚 Academic expenses like printing, stationery, and lab fees
- 📅 The month of the year — because October and Diwali hit different 💀

The model learns from 300 months of realistic student spending data and predicts total monthly expense with an average error of just **₹561** — accurate enough to actually plan your month around.

---

## 🗂️ Project Structure

```
📁 expense-predictor/
│
├── 📄 main.py                        ← Run this. Does everything.
├── 📄 requirements.txt               ← All Python dependencies
├── 📄 README.md                      ← You are here 👋
├── 📄 setup.sh                       ← One command setup + run
│
├── 📁 data/
│   └── 📊 student_expenses.csv       ← Auto-generated dataset (300 rows)
│
└── 📁 outputs/
    ├── 📈 model_comparison.png        ← All 5 models compared visually
    ├── 🎯 actual_vs_predicted.png     ← How close predictions are
    ├── 🌟 feature_importance.png      ← What features matter most
    ├── 🔥 correlation_heatmap.png     ← How all features relate
    ├── 📅 seasonal_pattern.png        ← Spending pattern by month
    └── 📋 model_comparison.csv        ← Raw metrics in CSV format
```

---

## ⚙️ Setup and Installation

No complicated setup. No GUI needed. Just Python and a terminal. 💻

### ✅ Prerequisites
- Python 3.8 or above
- pip
- A terminal — Command Prompt, PowerShell, or bash

### 🚀 Quick Start — 3 Commands

```bash
# Step 1 — Clone the repository
git clone https://github.com/<your-username>/expense-predictor.git
cd expense-predictor

# Step 2 — Install dependencies
pip install -r requirements.txt

# Step 3 — Run the full pipeline
python main.py
```

### ⚡ Even Quicker — Just 1 Command

```bash
bash setup.sh
```

That is it. The script installs everything and runs the full pipeline automatically. All outputs land in the `outputs/` folder. ✅

---

## 📊 The Dataset

Since real personal finance data is private and hard to get, I built a **synthetic dataset** that reflects realistic Indian hostel student spending patterns. 🇮🇳

Every parameter — allowance range, rent distribution, food as a percentage of allowance — was chosen to match what actually feels true for students living away from home in India.

### 🔢 Dataset at a Glance

| Property | Value |
|----------|-------|
| 📏 Total Records | 300 monthly entries |
| 📐 Input Features | 8 features |
| 🎯 Target Variable | total_expense |
| 💵 Avg Monthly Expense | ₹8,201 |
| 📉 Min Expense | ₹4,991 |
| 📈 Max Expense | ₹12,206 |

### 🏷️ Feature Descriptions

| Feature | Description | Why Included |
|---------|-------------|--------------|
| `month` | Calendar month 1–12 | Captures seasonal spikes 📅 |
| `allowance` | Monthly money from family | Primary income source 💰 |
| `hostel_rent` | PG or hostel rent | Biggest fixed cost 🏠 |
| `food` | Canteen and outside food | Most variable expense 🍱 |
| `recharge` | Mobile and internet bills | Hits every single month 📱 |
| `transport` | Auto, bus, cab costs | Varies with how much you travel 🚌 |
| `entertainment` | Outings, movies, snacks | The fun spending 🎬 |
| `academic` | Printing, lab, stationery | College specific cost 📚 |
| `total_expense` | **Target variable** | What we are predicting 🎯 |

### 🎄 Seasonal Patterns Built In

The dataset includes realistic spending boosts that match the Indian student calendar:

- 🪔 **October and November** — Diwali, festival shopping, travelling back home
- 📖 **January** — New semester starts, books, lab fees, assignment printing

---

## 🤖 The Machine Learning Models

Five regression models were trained and compared — from the simplest possible baseline to advanced ensemble methods. The goal was not just to get one good result but to understand *why* some models work better than others. 🏆

### 1️⃣ Linear Regression
The most basic model. Assumes a straight-line relationship between features and total expense. Useful as a **baseline** to judge if fancier models are actually adding value.

### 2️⃣ Ridge Regression
Linear Regression with **L2 regularisation** added. This helps when features are correlated with each other — like allowance and food, since food spending often scales with how much money you have. Prevents the model from over-relying on any one feature. 🛡️

### 3️⃣ Decision Tree Regressor
A non-linear model that splits data into branches based on feature thresholds. Easy to visualise and understand, but tends to **overfit** without any ensembling applied. 🌳

### 4️⃣ Random Forest Regressor
Builds **100 decision trees** and averages their predictions. Much more stable and reliable than a single tree. Also gives us feature importance scores as a useful bonus output. 🌲🌲🌲

### 5️⃣ Gradient Boosting Regressor ⭐ Best Model
Builds trees **sequentially** where each new tree learns from the mistakes of the previous one. This is what gave us the best results across all metrics. Smart, powerful, and the clear winner. 🚀

---

## 📈 Results

### 🏆 Model Comparison Table

| Model | MAE ₹ | RMSE ₹ | R² Score | CV R² |
|-------|--------|---------|----------|-------|
| Linear Regression | 596 | 750 | 0.6948 | 0.8195 |
| Ridge Regression | 592 | 755 | 0.6903 | 0.8186 |
| Decision Tree | 872 | 1099 | 0.3442 | 0.5407 |
| Random Forest | 639 | 812 | 0.6418 | 0.7474 |
| **Gradient Boosting** ⭐ | **561** | **699** | **0.7349** | **0.8092** |

### 🧠 What These Numbers Actually Mean

- **MAE ₹561** — On average the model is only ₹561 off per month. For a student spending around ₹8,200 that is less than 7% error 💪
- **RMSE ₹699** — Similar to MAE but penalises big prediction mistakes more heavily. Still very solid.
- **R² = 0.73** — The model explains 73% of why students spend differently from each other
- **CV R² = 0.80** — Results were tested across 5 different data splits and held up consistently. Not a fluke. 🎲

---

## 📉 Output Visualisations

### 📊 Model Comparison
*All five models compared across MAE, RMSE, and R² — Gradient Boosting clearly on top*

![Model Comparison](outputs/model_comparison.png)

---

### 🎯 Actual vs Predicted Expenses
*Each dot is one month of data. Dots closer to the red line mean more accurate predictions*

![Actual vs Predicted](outputs/actual_vs_predicted.png)

---

### 🌟 Feature Importance
*Which features the Random Forest relied on most — allowance and rent dominate*

![Feature Importance](outputs/feature_importance.png)

---

### 🔥 Correlation Heatmap
*How every feature relates to every other feature and to total expense*

![Correlation Heatmap](outputs/correlation_heatmap.png)

---

### 📅 Seasonal Spending Pattern
*Average total expense across all 12 months — notice the October/November Diwali spike and the January semester bump*

![Seasonal Pattern](outputs/seasonal_pattern.png)

---

## 💡 Key Findings

After running all five models and going through the outputs, here is what actually stood out: 🔍

- 🏆 **Gradient Boosting wins clearly** — beats Decision Tree by almost 0.40 in R². That is not a small gap.
- 📱 **Recharge is the most consistent expense** — small amount but shows up every single month like clockwork. Reliable predictor.
- 🪔 **Seasonality is real and measurable** — October and November spending is noticeably higher in the data, confirming the festival spike is not just a feeling
- 🏠 **Rent and allowance are the strongest predictors** — makes total sense since most other spending naturally scales with how much money you have
- 🌳 **Single Decision Tree fails without ensembling** — CV R² of 0.54 versus test R² of 0.34 shows clear overfitting
- 📊 **Linear models actually held up surprisingly well** — R² of 0.69 suggests student spending really does follow fairly linear patterns relative to income

---

## 🧗 Challenges I Faced

Building this from scratch was not entirely smooth. A few things genuinely tripped me up: 😅

**💻 matplotlib crashing on headless terminal**
Kept getting display errors when running from the command line. Turned out I needed `matplotlib.use("Agg")` at the very top before any other matplotlib import. Simple fix but took time to figure out.

**📉 Entertainment values going negative**
Early versions had entertainment spending going negative for low-allowance students because of random noise I added. Fixed it with `.clip(0)` — learned about the importance of sanity checking generated data.

**🤔 Whether to scale features for tree models**
Decision trees and random forests work with feature splits not distances so they do not need StandardScaler. Linear models do. Had to apply scaling selectively only to linear and ridge regression.

**📅 Getting the seasonal months right**
I initially put the festival spike in December like most Western datasets. Then I realised for Indian students the real expensive months are October/November for Diwali and January for new semester — completely different calendar.

---

## 🔮 What I Would Add Next

If I had more time, here is where this project goes next: 🛠️

- 🖥️ **CLI prediction tool** — type in your own numbers and get a prediction instantly without any coding
- 🌐 **Streamlit web app** — so friends in the hostel can actually use it without touching Python
- 📊 **Real data collection** — gather actual spending data from students with their permission
- 💾 **Savings predictor output** — not just total expense but how much of the allowance is likely to be left over
- 📆 **Time series modelling** — ARIMA or Prophet to properly capture the month-to-month trends
- 🔔 **Budget alert** — flag months that are predicted to be unusually expensive so you can prepare early

---

## 📦 Dependencies

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
```

Install everything at once:

```bash
pip install -r requirements.txt
```

---

## 🧪 Reproducing the Results

Everything uses `random_state=42` so results are fully reproducible every time. 🎲

```bash
python main.py

# What you should see:
# ✅ data/student_expenses.csv — 300 rows, 9 columns
# ✅ outputs/model_comparison.png
# ✅ outputs/actual_vs_predicted.png
# ✅ outputs/feature_importance.png
# ✅ outputs/correlation_heatmap.png
# ✅ outputs/seasonal_pattern.png
# ✅ Best model: Gradient Boosting — R²=0.73, MAE=₹561
```

---

## 📄 License

MIT License — free to use, modify, and share. 🆓

---

## 🙏 Acknowledgements

- 🔬 **Scikit-learn** — for making ML accessible without writing everything from scratch
- 🐼 **Pandas and NumPy** — the backbone of every data project
- 📊 **Matplotlib and Seaborn** — for making the results actually look good
- 🏫 **Course: Fundamentals of AI and ML** — for giving me the foundation to build this
- 🍽️ **My hostel canteen** — for being expensive enough to inspire this entire project 😂

---

<div align="center">

### 💬 *"I built this because I was tired of being broke on the 20th. If it helps you plan better too — even better."* 🚀

⭐ **If you found this useful, drop a star on the repo!**

*Made with 💙 by a hostel student who finally understands where the money goes*

</div>
