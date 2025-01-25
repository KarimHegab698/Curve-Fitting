import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(42)
number_points = int(input("Enter Number of points:"))
x = np.linspace(0, 10, number_points)
true_signal = np.cos(x)
noise = np.random.normal(0, 0.2, number_points)
noisy_signal = true_signal + noise

data = pd.DataFrame({"x": x, "noisy_signal": noisy_signal})
if os.path.exists("noisy_sinusoidal_data.csv"):
    data.to_csv("noisy_sinusoidal_data.csv", index=False)
#data = pd.DataFrame({"x": x, "true_signal": true_signal})
#data.to_csv("true_sinusoidal_data.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(x, true_signal, label="True Signal", linestyle='--')
plt.scatter(x, noisy_signal, label="Noisy Signal", color='Lime', s=10, alpha=0.7)
plt.title("Noisy Sinusoidal Data")
plt.xlabel("x")
plt.ylabel("Signal")
plt.legend()
plt.grid(True)
plt.show()