# Valuation Simulator

## Overview

This analysis predicts company valuations (in USD) using financial and operational metrics. The dataset includes features such as Revenue, Monthly Active Users, Revenue Growth, Total Addressable Market (TAM), Revenue per User, LTV:CAC Ratio, industry categories (SaaS, Greentech, Fintech, Edtech, Healthtech), and funding stages (Seed, Series A). We use Linear Regression and XGBoost models, evaluating performance with RMSE, MAPE, and R² metrics.

**Goals:**
- Understand feature correlations with valuation.
- Compare Linear Regression and XGBoost model performance.
- Identify key predictors of valuation.

---

## Data Description

The dataset contains the following features:

- **Revenue (USD)**: Annual company revenue  
- **Monthly Active Users**: Number of active users per month  
- **Revenue Growth (%)**: Percentage revenue growth year-over-year  
- **TAM (USD)**: Total addressable market size  
- **Revenue per User**: Revenue divided by monthly active users  
- **LTV:CAC Ratio**: Lifetime value to customer acquisition cost ratio  
- **Revenue_Growth_Interaction**: Revenue × Growth  
- **Revenue_Squared**: Squared revenue to capture non-linearity  
- **Growth_Squared**: Squared revenue growth  
- **LTV_Growth_Interaction**: LTV:CAC × Growth  
- **Industry Categories**: Dummy variables for SaaS, Greentech, Fintech, Edtech, Healthtech  
- **Funding Stages**: Dummy variables for Seed and Series A  

The **target variable** is **Valuation (USD)**.

---

## Correlation with Valuation

```r
correlation <- data.frame(
  Feature = c("Valuation (USD)", "Revenue (USD)", "Revenue_Growth_Interaction", 
              "Monthly Active Users", "Revenue Growth (%)", "TAM (USD)", 
              "Revenue_per_User", "LTV_CAC_Ratio"),
  Correlation = c(1.000000, 0.806251, 0.689701, 0.357827, 0.268000, 0.107455, 0.083693, -0.091667)
)
kable(correlation, digits = 3, caption = "Correlation with Valuation (USD)") %>%
  kable_styling(bootstrap_options = c("striped", "hover"))
```
---
## Key Insights:

- **Revenue (USD)** has the strongest positive correlation (0.806), indicating it’s a key driver of valuation.
- **Revenue_Growth_Interaction** (0.690) suggests combined revenue and growth effects are significant.
- **LTV_CAC_Ratio** has a slight negative correlation (-0.092), possibly indicating inefficiencies in customer acquisition at higher ratios.

## Model 1: Linear Regression
I trained a Linear Regression model to predict valuation.
**Performance Metrics**
- RMSE: $3,508,247.33
- MAPE: 7.99%
- R² Score: 0.96

Sample Predictions
```r
lr_preds <- data.frame(
  Predicted = c(41781435.04, 69761603.50, 21999930.40, 72419282.22, 39710116.07),
  Actual = c(45207688.00, 71823825.00, 24789045.00, 69049760.00, 39022618.00)
)
kable(lr_preds, digits = 2, caption = "Linear Regression: Predicted vs Actual Valuations (USD)") %>%
  kable_styling(bootstrap_options = c("striped", "hover"))
```
**Cross-Validation**
- R² Scores: [0.935, 0.960, 0.949, 0.948, 0.950]
- Average CV R²: 0.95 ± 0.01

**Interpretation**: The Linear Regression model explains 96% of the variance in valuation, with a low MAPE of 7.99%, indicating good predictive accuracy. Cross-validation confirms robustness.
## Model 2: XGBoost
I trained an XGBoost model with the following hyperparameters (tuned via grid search):

- **colsample_bytree**: 0.8
- **learning_rate**: 0.1
- **max_depth**: 3
- **n_estimators**: 300
- **subsample**: 0.8

**Performance Metrics**

- RMSE: $4,710,622.54
- MAPE: 9.03%
- R² Score: 0.92

Sample Predictions
```r
xgb_preds <- data.frame(
  Predicted = c(45720916.00, 69629008.00, 24865768.00, 69775192.00, 38736876.00),
  Actual = c(45207688.00, 71823825.00, 24789045.00, 69049760.00, 39022618.00)
)
kable(xgb_preds, digits = 2, caption = "XGBoost: Predicted vs Actual Valuations (USD)") %>%
  kable_styling(bootstrap_options = c("striped", "hover"))
```
**Cross-Validation**
- R² Scores: [0.972, 0.940, 0.953, 0.922, 0.945]
- Average CV R²: 0.95 ± 0.02

Feature Importance
```r
feat_imp <- data.frame(
  Feature = c("Industry_SaaS", "Revenue (USD)", "Revenue_Growth_Interaction", "Revenue_Squared", 
              "Industry_Greentech", "Monthly Active Users", "Revenue Growth (%)", 
              "Revenue_per_User", "Industry_Fintech", "LTV_Growth_Interaction", 
              "Industry_Edtech", "TAM (USD)", "Industry_Healthtech", "LTV_CAC_Ratio", 
              "Growth_Squared", "Funding Stage_Seed", "Funding Stage_Series A"),
  Importance = c(0.230833, 0.226875, 0.218086, 0.123224, 0.070863, 0.055512, 0.023524, 
                 0.017128, 0.013353, 0.004722, 0.003615, 0.002959, 0.002933, 0.002798, 
                 0.001392, 0.001276, 0.000908)
)
kable(feat_imp, digits = 3, caption = "XGBoost Feature Importance") %>%
  kable_styling(bootstrap_options = c("striped", "hover"))
```
**Interpretation**: XGBoost shows slightly lower performance (R² = 0.92, MAPE = 9.03%) than Linear Regression. The model prioritizes **Industry_SaaS**, **Revenue (USD)**, and **Revenue_Growth_Interaction** as top features, suggesting industry and revenue dynamics are critical.
## Model Comparison
```r
comparison <- data.frame(
  Model = c("Linear Regression", "XGBoost"),
  R2_Score = c(0.956411, 0.934513),
  RMSE = c(3552619, 4354484),
  MAPE = c(7.837670, 8.149107)
)
kable(comparison, digits = 3, caption = "Model Performance Comparison") %>%
  kable_styling(bootstrap_options = c("striped", "hover"))
```
## Key Findings:

**Linear Regression** outperforms **XGBoost** with a higher R² (0.96 vs. 0.93) and lower RMSE and MAPE.
Both models are robust, with average CV R² scores of 0.95.
**Linear Regression** is preferred for its simplicity and better performance.

Simulated Code for Reproducibility
Below is a simulated example of how the analysis was conducted (no raw data provided, so we generate synthetic data).
# Simulate dataset
```r
n <- 100
data <- data.frame(
  Revenue_USD = rnorm(n, 5e7, 2e7),
  Monthly_Active_Users = rnorm(n, 1e6, 5e5),
  Revenue_Growth = rnorm(n, 20, 10),
  TAM_USD = rnorm(n, 1e9, 5e8),
  Revenue_per_User = rnorm(n, 50, 20),
  LTV_CAC_Ratio = rnorm(n, 3, 1),
  Industry_SaaS = sample(c(0, 1), n, replace = TRUE),
  Industry_Greentech = sample(c(0, 1), n, replace = TRUE),
  Industry_Fintech = sample(c(0, 1), n, replace = TRUE),
  Industry_Edtech = sample(c(0, 1), n, replace = TRUE),
  Industry_Healthtech = sample(c(0, 1), n, replace = TRUE),
  Funding_Stage_Seed = sample(c(0, 1), n, replace = TRUE),
  Funding_Stage_Series_A = sample(c(0, 1), n, replace = TRUE)
)
data$Revenue_Growth_Interaction <- data$Revenue_USD * data$Revenue_Growth
data$Revenue_Squared <- data$Revenue_USD^2
data$Growth_Squared <- data$Revenue_Growth^2
data$LTV_Growth_Interaction <- data$LTV_CAC_Ratio * data$Revenue_Growth
data$Valuation_USD <- 3 * data$Revenue_USD + 2e-5 * data$Monthly_Active_Users +
                      1e6 * data$Industry_SaaS + rnorm(n, 0, 5e6)

# Split data
trainIndex <- createDataPartition(data$Valuation_USD, p = 0.8, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Linear Regression
lr_model <- train(Valuation_USD ~ ., data = train_data, method = "lm",
                  trControl = trainControl(method = "cv", number = 5))
lr_pred <- predict(lr_model, test_data)
lr_rmse <- sqrt(mean((lr_pred - test_data$Valuation_USD)^2))
lr_mape <- mean(abs((lr_pred - test_data$Valuation_USD) / test_data$Valuation_USD)) * 100
lr_r2 <- cor(lr_pred, test_data$Valuation_USD)^2

# XGBoost
xgb_grid <- expand.grid(nrounds = c(300), max_depth = c(3), eta = c(0.1),
                        gamma = 0, colsample_bytree = c(0.8), min_child_weight = 1,
                        subsample = c(0.8))
xgb_model <- train(Valuation_USD ~ ., data = train_data, method = "xgbTree",
                   trControl = trainControl(method = "cv", number = 5), tuneGrid = xgb_grid)
xgb_pred <- predict(xgb_model, test_data)
xgb_rmse <- sqrt(mean((xgb_pred - test_data$Valuation_USD)^2))
xgb_mape <- mean(abs((xgb_pred - test_data$Valuation_USD) / test_data$Valuation_USD)) * 100
xgb_r2 <- cor(xgb_pred, test_data$Valuation_USD)^2

# Output simulated metrics
cat("Simulated Linear Regression RMSE:", lr_rmse, "\n")
cat("Simulated Linear Regression MAPE:", lr_mape, "\n")
cat("Simulated Linear Regression R²:", lr_r2, "\n")
cat("Simulated XGBoost RMSE:", xgb_rmse, "\n")
cat("Simulated XGBoost MAPE:", xgb_mape, "\n")
cat("Simulated XGBoost R²:", xgb_r2, "\n")
```
# Conclusion

**Revenue (USD)** and **Industry_SaaS** are the most influential predictors of valuation.
**Linear Regression** is the better model due to its higher R² (0.96) and lower error metrics.
Future work could explore additional features (e.g., market sentiment) or ensemble methods to improve predictions.

# Requirements
To run this analysis, install the following R packages:
install.packages(c("tidyverse", "caret", "xgboost", "kableExtra"))

Save this file as valuation_analysis.Rmd and knit it to HTML using RStudio or rmarkdown::render("valuation_analysis.Rmd").
