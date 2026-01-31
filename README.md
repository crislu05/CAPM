# CAPM Analysis Project

A comprehensive Capital Asset Pricing Model (CAPM) analysis tool that performs rolling regression analysis on multiple stocks to calculate beta coefficients, R-squared values, and expected returns.

## Overview

This project analyzes 10 major stocks using the CAPM framework:
- **AAPL** (Apple)
- **JPM** (JPMorgan Chase)
- **XOM** (Exxon Mobil)
- **JNJ** (Johnson & Johnson)
- **WMT** (Walmart)
- **NEE** (NextEra Energy)
- **TSLA** (Tesla)
- **CAT** (Caterpillar)
- **NFLX** (Netflix)
- **LMT** (Lockheed Martin)

## Features

- **Rolling CAPM Regression**: Calculates beta and R² using a 126-day (6-month) rolling window
- **Static CAPM Analysis**: Full-sample regression for each stock
- **Visualization**: 
  - Individual stock regression plots
  - Rolling beta and R² time series
  - COVID-19 period comparison (crisis vs recovery vs normal regimes)
  - Collective plots for all stocks
- **Statistical Analysis**: 
  - Alpha and beta coefficients with p-values
  - R-squared values
  - Standard errors and confidence intervals
  - Expected vs realized returns

## Requirements

```bash
pip install yfinance pandas numpy scipy matplotlib
```

## Usage

Simply run the main script:

```bash
python main.py
```

The script will:
1. Download 10 years of historical price data for all stocks and the S&P 500
2. Calculate daily returns
3. Perform rolling CAPM regressions
4. Generate all visualizations in the `plots/` directory
5. Print summary statistics and expected return comparisons

## Output

The script generates three types of plots:

1. **Regression Plots** (`plots/regression/`): Linear regression scatter plots for the first 126-day window
2. **Rolling Analysis** (`plots/rolling/`): Time series of rolling beta and R² values
3. **Comparison Plots** (`plots/comparison/`): Bar charts comparing beta and R² across different market regimes

## Market Regimes Analyzed

- **COVID Crisis Regime**: February 24, 2020 - April 30, 2020
- **Post Crisis Recovery Regime**: May 1, 2020 - June 30, 2021
- **Normal Market Regime**: All other periods

## Methodology

The CAPM regression model used is:
```
R_i = α + β * R_m + ε
```

Where:
- `R_i` = Stock return
- `R_m` = Market return (S&P 500)
- `α` = Alpha (excess return)
- `β` = Beta (systematic risk)
- `ε` = Error term

The rolling window approach uses 126 trading days (approximately 6 months) to calculate time-varying beta and R² values.

