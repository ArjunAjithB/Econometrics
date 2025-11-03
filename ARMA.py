# Basic ARMA example in Python
# Requirements: statsmodels, numpy, matplotlib, pandas
# Install if needed: pip install statsmodels matplotlib pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# -----------------------
# 1) Generate synthetic ARMA(2,1) data
# -----------------------
np.random.seed(42)
n = 500

# AR and MA coefficients for the process (statsmodels uses the sign convention with leading 1)
# For AR: 1 - phi1*L - phi2*L^2  -> so pass [1, -phi1, -phi2]
ar_coefs = np.array([1, -0.6, 0.2])   # corresponds to phi1=0.6, phi2=-0.2
ma_coefs = np.array([1, 0.5])         # corresponds to theta1=0.5

y = arma_generate_sample(ar=ar_coefs, ma=ma_coefs, nsample=n, scale=1.0)

# make a pandas Series with a datetime index (optional but useful)
idx = pd.date_range(start="2000-01-01", periods=n, freq="D")
y = pd.Series(y, index=idx)

# quick plot of synthetic data
y.plot(title="Synthetic ARMA(2,1) series", figsize=(10,3))
plt.show()

# -----------------------
# 2) Fit ARMA via ARIMA (d=0)
# -----------------------
# choose orders p and q (here we know true is p=2, q=1)
p, d, q = 2, 0, 1
model = ARIMA(y, order=(p, d, q))
res = model.fit()

print(res.summary())

# -----------------------
# 3) Inspect residuals & diagnostics
# -----------------------
residuals = res.resid

fig, axes = plt.subplots(3, 1, figsize=(10,8), sharex=True)
axes[0].plot(residuals)
axes[0].set_title("Residuals")
sm.graphics.tsa.plot_acf(residuals, ax=axes[1], lags=40, title="ACF of residuals")
sm.graphics.tsa.plot_pacf(residuals, ax=axes[2], lags=40, title="PACF of residuals")
plt.tight_layout()
plt.show()

# Ljung-Box test for residual autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
ljung_box = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print("\nLjung-Box test (residuals):\n", ljung_box)

# -----------------------
# 4) In-sample fitted values vs actual
# -----------------------
fitted = res.predict(start=y.index[0], end=y.index[-1])
plt.figure(figsize=(10,3))
plt.plot(y, label="Actual")
plt.plot(fitted, label="Fitted", alpha=0.8)
plt.legend()
plt.title("Actual vs Fitted")
plt.show()

# -----------------------
# 5) Forecasting
# -----------------------
steps = 20
fc = res.get_forecast(steps=steps)
fc_mean = fc.predicted_mean
fc_ci = fc.conf_int()

plt.figure(figsize=(10,4))
plt.plot(y[-100:], label="Recent data (last 100)")
plt.plot(fc_mean.index, fc_mean, label="Forecast", marker='o')
plt.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.25)
plt.legend()
plt.title(f"{steps}-step forecast")
plt.show()

# Print forecast values
print("\nForecast (mean):")
print(fc_mean)
