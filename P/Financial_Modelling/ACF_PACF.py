import numpy as np
import matplotlib.pyplot as plt


acf_vals = np.array([0.42, 0.104, 0.032, -0.206, -0.138, 0.042, -0.018, 0.074])
pacf_vals = np.array([0.632, 0.381, 0.268, 0.199, 0.205, 0.101, 0.096,0.082])
lags = np.arange(1, 9)

# # --- parameters ---
# csv_file = "acf_pacf.csv"   # change if needed
# n = 100                     # sample size used to compute acf/pacf
# conf = 1.96 / np.sqrt(n)    # 95% CI

# # --- read data ---
# df = pd.read_csv(csv_file)

# lags = df["lag"].values
# acf_vals = df["acf"].values
# pacf_vals = df["pacf"].values


n = 100
conf = 1.96 / np.sqrt(n)

fig, ax = plt.subplots()

ax.vlines(lags, 0, acf_vals, colors='black')
ax.axhline(0, color='black')
ax.axhline(conf, color='blue', linestyle='--')
ax.axhline(-conf, color='blue', linestyle='--')
ax.set_xlabel('Lag')
ax.set_ylabel('ACF')
ax.set_xticks(lags)
ax.set_ylim(-0.25, 0.5)
plt.show()


fig, ax = plt.subplots()
ax.vlines(lags, 0, pacf_vals, colors='black')
ax.axhline(0, color='black')
ax.axhline(conf, color='blue', linestyle='--')
ax.axhline(-conf, color='blue', linestyle='--')
ax.set_xlabel('Lag')
ax.set_ylabel('Partial ACF')
ax.set_xticks(lags)
plt.show()
