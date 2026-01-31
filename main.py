import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import glob

# =============================
# 1. Configuration
# =============================
STOCKS = ["AAPL", "JPM", "XOM", "JNJ", "WMT", "NEE", "TSLA", "CAT", "NFLX", "LMT"]
MARKET_TICKER = "^GSPC"
ROLLING_WINDOW = 126  # 6 months ≈ 126 trading days

end_date = datetime.now()
start_date = end_date - timedelta(days=10 * 365)

# =============================
# 2. Download price data
# =============================
tickers = STOCKS + [MARKET_TICKER]

prices = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)["Close"]

# =============================
# 3. Compute daily returns
# =============================
returns = prices.pct_change().dropna()

market_returns = returns[MARKET_TICKER]

# =============================
# 4. Build CAPM datasets
# =============================
capm_data = {}

for ticker in STOCKS:
    df = pd.DataFrame({
        "stock_return": returns[ticker],
        "market_return": market_returns
    }).dropna()

    capm_data[ticker] = df

# =============================
# 5. CAPM regression function
# =============================
def run_capm(df):
    x = df["market_return"].values
    y = df["stock_return"].values
    
    # Add constant for intercept
    X = np.column_stack([np.ones(len(x)), x])
    
    # OLS using normal equation
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha, beta = coeffs[0], coeffs[1]
    
    # Calculate residuals and R²
    y_pred = alpha + beta * x
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate standard errors and p-values
    n = len(x)
    mse = ss_res / (n - 2)
    var_coeffs = mse * np.linalg.inv(X.T @ X)
    se_alpha = np.sqrt(var_coeffs[0, 0])
    se_beta = np.sqrt(var_coeffs[1, 1])
    
    t_alpha = alpha / se_alpha
    t_beta = beta / se_beta
    pval_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), n - 2))
    pval_beta = 2 * (1 - stats.t.cdf(abs(t_beta), n - 2))
    
    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "alpha_pvalue": pval_alpha,
        "beta_pvalue": pval_beta,
        "se_alpha": se_alpha,
        "se_beta": se_beta
    }

# =============================
# 6. Rolling CAPM regression
# =============================
def rolling_capm(df, window=126):
    alphas = []
    betas = []
    r2s = []
    beta_pvals = []
    se_betas = []
    dates = []

    for i in range(window, len(df)):
        window_df = df.iloc[i - window:i]
        x = window_df["market_return"].values
        y = window_df["stock_return"].values
        
        # Add constant for intercept
        X = np.column_stack([np.ones(len(x)), x])
        
        # OLS using normal equation
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha, beta = coeffs[0], coeffs[1]
        
        # Calculate R²
        y_pred = alpha + beta * x
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate standard error and p-value for beta
        n = len(x)
        if n > 2:
            mse = ss_res / (n - 2)
            var_coeffs = mse * np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(var_coeffs[1, 1])
            t_beta = beta / se_beta if se_beta > 0 else 0
            pval_beta = 2 * (1 - stats.t.cdf(abs(t_beta), n - 2))
        else:
            se_beta = np.nan
            pval_beta = 1.0

        alphas.append(alpha)
        betas.append(beta)
        r2s.append(r_squared)
        beta_pvals.append(pval_beta)
        se_betas.append(se_beta)
        dates.append(df.index[i])

    return pd.DataFrame(
        {
            "alpha": alphas,
            "beta": betas,
            "r_squared": r2s,
            "beta_pvalue": beta_pvals,
            "se_beta": se_betas
        },
        index=dates
    )

# =============================
# 7. Run rolling CAPM for all stocks
# =============================
rolling_results = {}

for ticker in STOCKS:
    rolling_results[ticker] = rolling_capm(
        capm_data[ticker],
        window=ROLLING_WINDOW
    )

# =============================
# 8. Example: static CAPM output
# =============================
print("Static CAPM regression for AAPL:")
result = run_capm(capm_data["AAPL"])
print(f"Alpha: {result['alpha']:.6f} (p-value: {result['alpha_pvalue']:.4f})")
print(f"Beta: {result['beta']:.4f} (p-value: {result['beta_pvalue']:.4f})")
print(f"R-squared: {result['r_squared']:.4f}")
print()

# =============================
# 9. Plot linear regression for first CAPM window
# =============================
# Close any existing figures to ensure fresh plots
plt.close('all')

# Delete all existing regression plots
regression_plots = glob.glob("plots/regression/*.png")
for plot_file in regression_plots:
    try:
        os.remove(plot_file)
    except:
        pass

os.makedirs("plots/regression", exist_ok=True)

for ticker in STOCKS:
    # Extract first window of data
    df = capm_data[ticker]
    if len(df) >= ROLLING_WINDOW:
        first_window_df = df.iloc[:ROLLING_WINDOW]
        
        # Run CAPM regression on first window
        result = run_capm(first_window_df)
        
        # Extract data for plotting
        x = first_window_df["market_return"].values
        y = first_window_df["stock_return"].values
        
        # Create regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = result["alpha"] + result["beta"] * x_line
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot
        ax.scatter(x, y, alpha=0.5, s=20)
        
        # Regression line
        ax.plot(x_line, y_line, 'r-', linewidth=2, 
                label=f"α: {result['alpha']:.6f}, β: {result['beta']:.4f}, "
                      f"R²: {result['r_squared']:.4f}")
        
        # Formatting
        ax.set_xlabel("Market Return")
        ax.set_ylabel("Stock Return")
        ax.set_title(
            f"CAPM Linear Regression — {ticker} (First {ROLLING_WINDOW} Days)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Overwrite existing file to ensure plot is updated
        plt.savefig(f"plots/regression/{ticker}_first_window_regression.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

# =============================
# 10. Plot rolling beta and R²
# =============================
# Close any existing figures to ensure fresh plots
plt.close('all')

# Delete all existing rolling plots
rolling_plots = glob.glob("plots/rolling/*.png")
for plot_file in rolling_plots:
    try:
        os.remove(plot_file)
    except:
        pass

os.makedirs("plots/rolling", exist_ok=True)

for ticker in STOCKS:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Get rolling results
    rolling_df = rolling_results[ticker]
    dates = rolling_df.index
    beta_values = rolling_df["beta"]
    se_beta_values = rolling_df["se_beta"]
    
    # Define regime periods
    crisis_start = pd.Timestamp('2020-02-24')
    crisis_end = pd.Timestamp('2020-04-30')
    recovery_start = pd.Timestamp('2020-05-01')
    recovery_end = pd.Timestamp('2021-06-30')
    
    # Calculate 95% confidence intervals for beta
    z_score = 1.96  # 95% confidence interval
    beta_upper = beta_values + z_score * se_beta_values
    beta_lower = beta_values - z_score * se_beta_values
    
    # Mark regime periods on both plots
    for ax in axes:
        ax.axvspan(crisis_start, crisis_end, alpha=0.15, color='orange', label='COVID Crisis Regime')
        ax.axvspan(recovery_start, recovery_end, alpha=0.15, color='green', label='Post Crisis Recovery Regime')
    
    # Plot uncertainty bands (shaded area) for beta (no label)
    axes[0].fill_between(
        dates,
        beta_lower,
        beta_upper,
        alpha=0.2,
        color='blue'
    )
    
    # Plot beta line using matplotlib directly (avoids converter conflict, no label)
    axes[0].plot(dates, beta_values, color='blue', linewidth=2)
    
    # Add reference line at beta = 1
    axes[0].axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='β = 1 (Market)')
    
    axes[0].set_title(f"β — {ticker}")
    axes[0].set_ylabel("Beta")
    axes[0].legend(loc='upper left')
    
    # Plot R² using matplotlib directly (avoids converter conflict)
    axes[1].plot(dates, rolling_df["r_squared"], color='blue', linewidth=2)
    axes[1].set_title(f"R² — {ticker}")
    axes[1].set_ylabel("R²")
    
    # Format x-axis to show years on both plots
    # Apply formatting after both plots are created
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', rotation=45)
    
    # Force top plot to show x-axis labels (sharex=True hides them by default)
    axes[0].tick_params(axis='x', labelbottom=True)

    plt.tight_layout()
    # Overwrite existing file to ensure plot is updated
    plt.savefig(f"plots/rolling/{ticker}_rolling_capm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# =============================
# 11. Collective plots for all stocks
# =============================
plt.close('all')

# Define regime periods
crisis_start = pd.Timestamp('2020-02-24')
crisis_end = pd.Timestamp('2020-04-30')
recovery_start = pd.Timestamp('2020-05-01')
recovery_end = pd.Timestamp('2021-06-30')

# Create collective beta plot with 10 subplots
fig, axes = plt.subplots(5, 2, figsize=(16, 20), sharex=False, sharey=False)
axes = axes.flatten()

# Plot beta for each stock in separate subplot
for i, ticker in enumerate(STOCKS):
    ax = axes[i]
    df = rolling_results[ticker]
    dates = df.index
    beta_values = df["beta"]
    se_beta_values = df["se_beta"]
    
    # Calculate 95% confidence intervals for beta
    z_score = 1.96
    beta_upper = beta_values + z_score * se_beta_values
    beta_lower = beta_values - z_score * se_beta_values
    
    # Plot uncertainty bands (shaded area)
    ax.fill_between(dates, beta_lower, beta_upper, alpha=0.2, color='blue')
    
    # Mark regime periods (only label once)
    if i == 0:
        ax.axvspan(crisis_start, crisis_end, alpha=0.15, color='orange', label='COVID Crisis Regime')
        ax.axvspan(recovery_start, recovery_end, alpha=0.15, color='green', label='Post Crisis Recovery Regime')
        ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='β = 1 (Market)')
    else:
        ax.axvspan(crisis_start, crisis_end, alpha=0.15, color='orange')
        ax.axvspan(recovery_start, recovery_end, alpha=0.15, color='green')
        ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot beta line
    ax.plot(dates, beta_values, color='blue', linewidth=2)
    ax.set_title(f"{ticker}", fontsize=11, fontweight='bold')
    ax.set_ylabel("β", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=9)

# Add single legend to the first subplot
axes[0].legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig("plots/rolling/all_stocks_beta.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Create collective R² plot with 10 subplots
fig, axes = plt.subplots(5, 2, figsize=(16, 20), sharex=False, sharey=False)
axes = axes.flatten()

# Plot R² for each stock in separate subplot
for i, ticker in enumerate(STOCKS):
    ax = axes[i]
    df = rolling_results[ticker]
    dates = df.index
    r2_values = df["r_squared"]
    
    # Mark regime periods (only label once)
    if i == 0:
        ax.axvspan(crisis_start, crisis_end, alpha=0.15, color='orange', label='COVID Crisis Regime')
        ax.axvspan(recovery_start, recovery_end, alpha=0.15, color='green', label='Post Crisis Recovery Regime')
    else:
        ax.axvspan(crisis_start, crisis_end, alpha=0.15, color='orange')
        ax.axvspan(recovery_start, recovery_end, alpha=0.15, color='green')
    
    # Plot R² line
    ax.plot(dates, r2_values, color='blue', linewidth=2)
    ax.set_title(f"{ticker}", fontsize=11, fontweight='bold')
    ax.set_ylabel("R²", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=9)

# Add single legend to the first subplot
axes[0].legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig("plots/rolling/all_stocks_r2.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# =============================
# 12. COVID Period Comparison: Beta and R²
# =============================
plt.close('all')

# Create folder for comparison plots
os.makedirs("plots/comparison", exist_ok=True)

# Delete existing comparison plots
comparison_plots = glob.glob("plots/comparison/*.png")
for plot_file in comparison_plots:
    try:
        os.remove(plot_file)
    except:
        pass

# Define regime periods
crisis_start = pd.Timestamp('2020-02-24')
crisis_end = pd.Timestamp('2020-04-30')
recovery_start = pd.Timestamp('2020-04-01')
recovery_end = pd.Timestamp('2021-06-30')

# Calculate averages and standard errors for each stock
crisis_beta_avg = []
recovery_beta_avg = []
rest_beta_avg = []
crisis_beta_se = []
recovery_beta_se = []
rest_beta_se = []
crisis_r2_avg = []
recovery_r2_avg = []
rest_r2_avg = []
tickers_list = []

for ticker in STOCKS:
    df = rolling_results[ticker]
    
    # Filter regime periods
    crisis_mask = (df.index >= crisis_start) & (df.index <= crisis_end)
    recovery_mask = (df.index >= recovery_start) & (df.index <= recovery_end)
    rest_mask = ~(crisis_mask | recovery_mask)
    
    crisis_data = df[crisis_mask]
    recovery_data = df[recovery_mask]
    rest_data = df[rest_mask]
    
    # Calculate averages and propagate errors correctly for crisis period
    if len(crisis_data) > 0:
        crisis_beta_avg.append(crisis_data["beta"].mean())
        se_beta_squared = crisis_data["se_beta"] ** 2
        crisis_beta_se.append(np.sqrt(se_beta_squared.mean() / len(crisis_data)))
        crisis_r2_avg.append(crisis_data["r_squared"].mean())
    else:
        crisis_beta_avg.append(np.nan)
        crisis_beta_se.append(0)
        crisis_r2_avg.append(np.nan)
    
    # Calculate averages and propagate errors correctly for recovery period
    if len(recovery_data) > 0:
        recovery_beta_avg.append(recovery_data["beta"].mean())
        se_beta_squared = recovery_data["se_beta"] ** 2
        recovery_beta_se.append(np.sqrt(se_beta_squared.mean() / len(recovery_data)))
        recovery_r2_avg.append(recovery_data["r_squared"].mean())
    else:
        recovery_beta_avg.append(np.nan)
        recovery_beta_se.append(0)
        recovery_r2_avg.append(np.nan)
    
    # Calculate averages and propagate errors correctly for rest of dataset
    if len(rest_data) > 0:
        rest_beta_avg.append(rest_data["beta"].mean())
        se_beta_squared = rest_data["se_beta"] ** 2
        rest_beta_se.append(np.sqrt(se_beta_squared.mean() / len(rest_data)))
        rest_r2_avg.append(rest_data["r_squared"].mean())
    else:
        rest_beta_avg.append(np.nan)
        rest_beta_se.append(0)
        rest_r2_avg.append(np.nan)
    
    tickers_list.append(ticker)

# Create Beta comparison plot with error bars (three groups)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(tickers_list))
width = 0.25

bars1 = ax.bar(x - width, crisis_beta_avg, width, yerr=crisis_beta_se,
               label='COVID Crisis Regime (Feb 2020 - Apr 2020)', 
               color='orange', alpha=0.7, capsize=5, error_kw={'elinewidth': 1.5})
bars2 = ax.bar(x, recovery_beta_avg, width, yerr=recovery_beta_se,
               label='Post Crisis Recovery Regime (May 2020 - Jun 2021)', 
               color='green', alpha=0.7, capsize=5, error_kw={'elinewidth': 1.5})
bars3 = ax.bar(x + width, rest_beta_avg, width, yerr=rest_beta_se,
               label='Normal Market Regime', 
               color='blue', alpha=0.7, capsize=5, error_kw={'elinewidth': 1.5})

ax.set_xlabel('Stock Ticker', fontsize=11)
ax.set_ylabel('Average β', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(tickers_list)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("plots/comparison/covid_beta_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Create R² comparison plot (three groups)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(tickers_list))
width = 0.25

bars1 = ax.bar(x - width, crisis_r2_avg, width, label='COVID Crisis Regime (Feb 2020 - Apr 2020)', 
               color='orange', alpha=0.7)
bars2 = ax.bar(x, recovery_r2_avg, width, label='Post Crisis Recovery Regime (May 2020 - Jun 2021)', 
               color='green', alpha=0.7)
bars3 = ax.bar(x + width, rest_r2_avg, width, label='Normal Market Regime', 
               color='blue', alpha=0.7)

ax.set_xlabel('Stock Ticker', fontsize=11)
ax.set_ylabel('Average R²', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(tickers_list)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("plots/comparison/covid_r2_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# =============================
# 13. Summary statistics table
# =============================
summary_stats = []

for ticker in STOCKS:
    df = rolling_results[ticker]
    summary_stats.append({
        "Ticker": ticker,
        "Mean Beta": df["beta"].mean(),
        "Beta Std": df["beta"].std(),
        "Mean R²": df["r_squared"].mean(),
        "% Beta Significant": (df["beta_pvalue"] < 0.05).mean() * 100
    })

summary_df = pd.DataFrame(summary_stats)
print(summary_df)

# =============================
# 14. CAPM Expected Return (Rf = 0)
# =============================

# Mean daily market return
mean_market_return = market_returns.mean()

expected_return_results = []

for ticker in STOCKS:
    # Full-sample CAPM beta
    capm_result = run_capm(capm_data[ticker])
    beta = capm_result["beta"]
    
    # CAPM expected return (Rf = 0)
    expected_return = beta * mean_market_return
    
    # Realised mean return
    realised_return = capm_data[ticker]["stock_return"].mean()
    
    expected_return_results.append({
        "Ticker": ticker,
        "Beta": beta,
        "Expected Return (CAPM)": expected_return,
        "Realised Mean Return": realised_return
    })

expected_return_df = pd.DataFrame(expected_return_results)
print("\nCAPM Expected vs Realised Returns (Rf = 0):")
print(expected_return_df)
