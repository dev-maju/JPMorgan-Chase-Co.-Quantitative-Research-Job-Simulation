# JPMorgan-Chase-Co.-Quantitative-Research-Job-Simulation
This repository contains my solutions to the JPMorgan Chase &amp; Co. Quantitative Research Job Simulation, covering multiple problem domains including commodity pricing, storage contract valuation, credit risk modeling, and optimal risk bucketing using dynamic programming.

## Task 1 – Natural Gas Price Modeling
- Built a trend + seasonality statistical model to estimate natural gas prices.

### Used historical monthly price data to:
  - Interpolate prices at arbitrary dates
  - Extrapolate prices one year into the future
- The model captures seasonal effects and long-term trends, providing a smooth forward price curve.

### Key concepts:
- Time series modeling
- Seasonality extraction
- Forward price estimation

## Task 2 – Natural Gas Storage Contract Valuation
- Developed a deterministic intrinsic valuation model for a gas storage contract.

### The model prices:
  - Multiple injection and withdrawal dates
  - Storage capacity and rate constraints
  - Storage costs
- Cash flows are evaluated using prices from the Task 1 price model.

### Key concepts:
  - Commodity storage economics
  - Cash-flow-based contract valuation
  - Constraint-aware pricing

## Task 3 – Credit Risk Modeling (Probability of Default & Expected Loss)
- Built predictive models to estimate Probability of Default (PD) for personal loans.

### Implemented:
- Gradient Boosting

### Computed Expected Loss (EL) using:


> EL=PD×Exposure×(1−Recovery Rate)

### Addressed:
  - Target leakage
  - Over-confidence in predictions
  - Model calibration and interpretability

### Key concepts:
  - Credit risk modeling
  - Classification and model comparison
  - Expected loss estimation

## Task 4 – Optimal FICO Score Bucketing (Quantization)
- Designed a general, data-driven approach to discretize FICO scores into categorical risk ratings.
- Implemented maximum likelihood–based quantization using dynamic programming.

### The algorithm:
  - Maximizes the log-likelihood of observed defaults
  - Produces monotonic, interpretable FICO buckets
  - Scales efficiently via pre-aggregation
  - Outputs a rating map where lower ratings correspond to better credit quality.

### Key concepts:
- Quantization
- Maximum likelihood estimation
- Dynamic programming
- Credit score bucketing

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Dynamic Programming & Statistical Modeling

## Key Takeaways
- Demonstrates applied quantitative research across commodities and credit risk.
- Emphasizes model correctness, interpretability, and robustness.
- Reflects industry-standard practices used by trading desks and risk teams.

## Notes
- The datasets provided as part of the simulation are assumed to be educational / synthetic.
- High model performance in some tasks reflects the structure of the provided data and is discussed explicitly in the analysis.
