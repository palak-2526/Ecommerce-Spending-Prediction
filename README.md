# Customer Spending Prediction using Multiple Linear Regression

A machine learning project that predicts the **yearly amount spent** by e-commerce customers from their app, website, and membership activity, using Multiple Linear Regression in Python.

## Problem Statement

An online clothing store sells through both a mobile app and a website. This project studies which customer behaviors are linked to higher yearly spending, and builds a model that predicts spending from those behaviors.

## Dataset

- **Name:** Ecommerce Customers (from Kaggle)
- **File:** `Ecommerce Customers.csv`
- **Size:** 500 customers, no missing values

| Column | Description |
|---|---|
| Email, Address, Avatar | Customer details (not used for modeling) |
| Avg. Session Length | Average in-store style advice session length (minutes) |
| Time on App | Time spent on the mobile app (minutes) |
| Time on Website | Time spent on the website (minutes) |
| Length of Membership | Number of years as a member |
| **Yearly Amount Spent** | **Target variable** (what the model predicts) |

## Approach

1. **Load and inspect** the data with Pandas (`info()`, `describe()`).
2. **Exploratory Data Analysis** with Seaborn: joint plot, pair plot, and regression plot to study how features relate to yearly spending.
3. **Train/test split:** 70% training, 30% testing (`random_state=42`).
4. **Train** a Linear Regression model with Scikit-learn and inspect the coefficient of each feature.
5. **Predict and evaluate** on the test set using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
6. **Residual analysis** with a distribution plot and a Q-Q plot to check that the errors are roughly normally distributed.

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- SciPy
- Google Colab / Jupyter Notebook

## Project Structure

```
.
├── ecommerce_spending_prediction.ipynb   # Main notebook
├── Ecommerce Customers.csv               # Dataset
└── README.md
```

## How to Run

**On Google Colab**
1. Open [Google Colab](https://colab.research.google.com) and upload `ecommerce_spending_prediction.ipynb`.
2. Upload `Ecommerce Customers.csv` using the Files panel on the left.
3. Click **Runtime → Run all**.

**On your computer**
1. Clone this repository and open the folder.
2. Install the libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
   ```
3. Start Jupyter, open the notebook, and run all cells:
   ```bash
   jupyter notebook
   ```

## Results

Evaluated on the test set (150 customers, 30% of the data):

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | 8.43 |
| Mean Squared Error (MSE) | 103.92 |
| Root Mean Squared Error (RMSE) | 10.19 |

**Model coefficients** (change in Yearly Amount Spent for a one-unit increase in each feature, other features held constant):

| Feature | Coefficient |
|---|---|
| Length of Membership | 61.67 |
| Time on App | 38.60 |
| Avg. Session Length | 25.72 |
| Time on Website | 0.46 |

**Key takeaways**
- Length of Membership has the strongest effect on yearly spending, followed by Time on App.
- Time on Website has almost no effect, so improving the mobile app is likely to matter more than improving the website.
- The prediction and residual plots are in the notebook.

## What I Learned

- Performing EDA to understand feature relationships before modeling
- Building and evaluating a regression model with Scikit-learn
- Interpreting model coefficients and checking residuals

## Author

**Palak Dwivedi**

[GitHub](https://github.com/palak-2526) | [LinkedIn](https://www.linkedin.com/in/palak-dwivedi-a355712a9/) | dpalak256@gmail.com
