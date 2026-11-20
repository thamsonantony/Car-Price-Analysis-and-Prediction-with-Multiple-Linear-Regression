# Car Price Analysis and Prediction with Multiple Linear Regression

Which features drive a car's price, how much does each one matter, and how well can a linear model predict the price of cars it hasn't seen?

This started as the Group 1 final project for BUAN 660 Applied Statistics at Sacred Heart University (course SS24BUAN660AB, June 2024). The analysis has since been reworked to fix the original model's problems: no held-out evaluation, unequal error variance, no categorical features, and untested interaction effects.

## Results

All three models are scored on the same 41 held-out cars (20% test split, `random_state=42`).

| Model | Test R² | RMSE | MAE | MAPE |
|---|---|---|---|---|
| Original: price ~ horsepower + curb weight + engine size | 0.821 | $3,761 | $2,722 | 21.6% |
| Ridge, all features (`car_price_ridge_variant.ipynb`) | 0.904 | $2,757 | $1,893 | 16.1% |
| **Final: log-price MLR (`car_price_mlr.ipynb`)** | **0.920** | **$2,513** | **$1,652** | **11.3%** |

Final model: `log(price) ~ horsepower + curbweight + brand tier + body type + horsepower × brand tier`

## Key findings

- **Brand tier is the biggest driver of price.** At a typical 95 hp, a luxury-brand car costs about 78% more than a comparable budget-brand car, and a mid-tier car about 24% more.
- **Horsepower matters less for luxury brands.** +10 hp adds about 4.5% to price for budget brands, 3.8% for mid-tier brands and 1.7% for luxury brands (interaction p = 0.012).
- **Curb weight** adds about 4.4% per 100 lb.
- **Body type:** convertibles cost about 21% more than comparable sedans; hatchbacks (−6%) and wagons (−10%) cost less.
- **Engine size** is strongly correlated with price, but it adds nothing once brand tier and weight are in the model: luxury brands simply tend to have bigger engines.

These are associations in observational data, not causal effects.

## Method

1. **Cleaning:** drop the ID column, check for missing values and duplicates, extract the brand from `CarName` and fix misspellings (27 raw labels after lower-casing become 22 brands).
2. **Exploration:** price distribution (right-skewed, skew 1.78), box plots by category and brand, correlation matrix.
3. **Train/test split first:** 80/20, so the final evaluation uses cars no model has seen.
4. **Baseline diagnostics:** the original three-feature model fails the constant-variance test (Breusch–Pagan p < 0.0001) and has non-normal residuals.
5. **Improvements:** log-price target, brand tier (budget / mid / luxury by brand median price, learned from training data only to avoid leakage), body type, and a horsepower × tier interaction tested with a partial F-test.
6. **Model selection:** 5-fold cross-validation repeated 5 times on the training set, with brand tiers re-learned inside each fold.
7. **Final checks:** HC3 robust standard errors, VIF, residual tests, Cook's distance, then one evaluation on the test set.

## Files

| File | Contents |
|---|---|
| `car_price_mlr.ipynb` | Main analysis: cleaning, EDA, baseline models, diagnostics, improved models, cross-validation and final evaluation |
| `car_price_ridge_variant.ipynb` | Prediction-focused companion: OLS, Ridge and Lasso on all features, compared with repeated cross-validation. Shows that the smaller log-price model in the main notebook predicts better |
| `CarPrice_Assignment.csv` | The dataset: 205 cars, 25 attributes plus an ID (from Kaggle) |

## How to run

Requires Python 3.10 or later.

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn jupyter
jupyter notebook car_price_mlr.ipynb
```

Run the notebooks from this folder, because they read `CarPrice_Assignment.csv` by relative path. `car_price_ridge_variant.ipynb` needs scikit-learn 1.4 or later (it uses `root_mean_squared_error`).

## Limitations

- 205 cars from the classic 1985 imports dataset, of which only 24 are luxury cars. The model is least accurate for that segment.
- The tier cut-offs ($12k and $20k brand median price) are a modelling choice. Other cut-offs would change the numbers somewhat.
- Residuals are still heavier-tailed than normal, so p-values near 0.05 should be read with care.

## Team

Original course project by Harika Naga Sai Dandu, Harshitha Kotichintala, Ruksar Sultana Shaik, Sai Keerthana Mallipeddi and Thamson Antony Arockiasamy. Instructor: Dr. Max K. Yun.
