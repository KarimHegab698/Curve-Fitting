import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from numpy.polynomial.polynomial import Polynomial

#Reading data from noisy file
data = pd.read_csv("noisy_sinusoidal_data.csv")
#data = pd.read_csv("weird_noisy_signal_dataset.csv")
x = data["x"].values
y = data["noisy_signal"].values
#y = data["y"].values

#Linear curve fitting
coefficients_linear = np.polyfit(x, y, 1)
y_linear_fit = np.polyval(coefficients_linear, x)

#Polynomial curve fitting 2nd,3rd degrees
coefficients_poly2 = np.polyfit(x, y, 2)
y_poly2nd_degree_fit = np.polyval(coefficients_poly2, x)

coefficients_poly3 = np.polyfit(x, y, 3)
y_poly3rd_degree_fit = np.polyval(coefficients_poly3, x)

coefficients_poly4 = np.polyfit(x, y, 4)
y_poly4th_degree_fit = np.polyval(coefficients_poly4, x)

coefficients_poly5 = np.polyfit(x, y, 5)
y_poly5th_degree_fit = np.polyval(coefficients_poly5, x)

coefficients_poly6 = np.polyfit(x, y, 6)
y_poly6th_degree_fit = np.polyval(coefficients_poly6, x)

coefficients_poly7 = np.polyfit(x, y, 7)
y_poly7th_degree_fit = np.polyval(coefficients_poly7, x)

coefficients_poly200 = np.polyfit(x, y, 200)
y_poly200th_degree_fit = np.polyval(coefficients_poly200, x)


#R^2 and MSE
mse_linear = mean_squared_error(y, y_linear_fit)
r2_linear = r2_score(y, y_linear_fit)

mse_poly2 = mean_squared_error(y, y_poly2nd_degree_fit)
r2_poly2 = r2_score(y, y_poly2nd_degree_fit)

mse_poly3 = mean_squared_error(y, y_poly3rd_degree_fit)
r2_poly3 = r2_score(y, y_poly3rd_degree_fit)

mse_poly4 = mean_squared_error(y, y_poly4th_degree_fit)
r2_poly4 = r2_score(y, y_poly4th_degree_fit)

mse_poly5 = mean_squared_error(y, y_poly5th_degree_fit)
r2_poly5 = r2_score(y, y_poly5th_degree_fit)

mse_poly6 = mean_squared_error(y, y_poly6th_degree_fit)
r2_poly6 = r2_score(y, y_poly6th_degree_fit)

mse_poly7 = mean_squared_error(y, y_poly7th_degree_fit)
r2_poly7 = r2_score(y, y_poly7th_degree_fit)

mse_poly200 = mean_squared_error(y, y_poly200th_degree_fit)
r2_poly200 = r2_score(y, y_poly200th_degree_fit)

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Noisy Data", color="Purple",s=10, alpha=0.7)
plt.plot(x, y_linear_fit, label=f"Linear Fit (R^2={r2_linear:.2f}, MSE={mse_linear:.2f})", color="red")
plt.title("Curve Fitting to Noisy Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Noisy Data", color="Purple",s=10, alpha=0.7)
plt.plot(x, y_poly2nd_degree_fit, label=f"Polynomial 2nd Degree Fit (R^2={r2_poly2:.2f}, MSE={mse_poly2:.2f})", color="blue")
plt.title("Curve Fitting to Noisy Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Noisy Data", color="Purple",s=10, alpha=0.7)
plt.plot(x, y_poly3rd_degree_fit, label=f"Polynomial 3rd Degree Fit (R^2={r2_poly3:.2f}, MSE={mse_poly3:.2f})", color="green")
plt.title("Curve Fitting to Noisy Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Noisy Data", color="Purple",s=10, alpha=0.7)
plt.plot(x, y_poly4th_degree_fit, label=f"Polynomial 4th Degree Fit (R^2={r2_poly4:.2f}, MSE={mse_poly4:.2f})", color="black")
plt.title("Curve Fitting to Noisy Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Noisy Data", color="Purple",s=10, alpha=0.7)
plt.plot(x, y_poly5th_degree_fit, label=f"Polynomial 5th Degree Fit (R^2={r2_poly5:.2f}, MSE={mse_poly5:.2f})", color="navy")
plt.title("Curve Fitting to Noisy Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Noisy Data", color="Purple",s=10, alpha=0.7)
plt.plot(x, y_poly6th_degree_fit, label=f"Polynomial 6th Degree Fit (R^2={r2_poly6:.2f}, MSE={mse_poly6:.2f})", color="yellow")
plt.title("Curve Fitting to Noisy Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Noisy Data", color="Purple",s=10, alpha=0.7)
plt.plot(x, y_poly7th_degree_fit, label=f"Polynomial 7th Degree Fit (R^2={r2_poly7:.2f}, MSE={mse_poly7:.2f})", color="cyan")
plt.title("Curve Fitting to Noisy Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Noisy Data", color="Purple",s=10, alpha=0.7)
plt.plot(x, y_poly200th_degree_fit, label=f"Polynomial 200th Degree Fit (R^2={r2_poly200:.2f}, MSE={mse_poly200:.2f})", color="orange")
plt.title("Curve Fitting to Noisy Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Noisy Data", color="Purple",s=10, alpha=0.7)
plt.plot(x, y_linear_fit, label=f"Linear Fit (R^2={r2_linear:.2f}, MSE={mse_linear:.2f})", color="red")
plt.plot(x, y_poly2nd_degree_fit, label=f"Polynomial 2nd Degree Fit (R^2={r2_poly2:.2f}, MSE={mse_poly2:.2f})", color="blue")
plt.plot(x, y_poly3rd_degree_fit, label=f"Polynomial 3rd Degree Fit (R^2={r2_poly3:.2f}, MSE={mse_poly3:.2f})", color="green")
plt.plot(x, y_poly4th_degree_fit, label=f"Polynomial 4th Degree Fit (R^2={r2_poly4:.2f}, MSE={mse_poly4:.2f})", color="black")
plt.plot(x, y_poly5th_degree_fit, label=f"Polynomial 5th Degree Fit (R^2={r2_poly5:.2f}, MSE={mse_poly5:.2f})", color="navy")
plt.plot(x, y_poly6th_degree_fit, label=f"Polynomial 6th Degree Fit (R^2={r2_poly6:.2f}, MSE={mse_poly6:.2f})", color="yellow")
plt.plot(x, y_poly7th_degree_fit, label=f"Polynomial 7th Degree Fit (R^2={r2_poly7:.2f}, MSE={mse_poly7:.2f})", color="cyan")
plt.plot(x, y_poly200th_degree_fit, label=f"Polynomial 200th Degree Fit (R^2={r2_poly200:.2f}, MSE={mse_poly200:.2f})", color="orange")
plt.title("Curve Fitting to Noisy Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()


#Display Results of R^2 And MSE
print("Linear Fit: R^2 = {:.2f}, MSE = {:.2f}".format(r2_linear, mse_linear))
print("Polynomial 2nd Degree Fit: R^2 = {:.2f}, MSE = {:.2f}".format(r2_poly2, mse_poly2))
print("Polynomial 3rd Degree Fit: R^2 = {:.2f}, MSE = {:.2f}".format(r2_poly3, mse_poly3))
print("Polynomial 4th Degree Fit: R^2 = {:.2f}, MSE = {:.2f}".format(r2_poly4, mse_poly4))
print("Polynomial 5th Degree Fit: R^2 = {:.2f}, MSE = {:.2f}".format(r2_poly5, mse_poly5))
print("Polynomial 6th Degree Fit: R^2 = {:.2f}, MSE = {:.2f}".format(r2_poly6, mse_poly6))
print("Polynomial 7th Degree Fit: R^2 = {:.2f}, MSE = {:.2f}".format(r2_poly7, mse_poly7))
print("Polynomial 200th Degree Fit: R^2 = {:.2f}, MSE = {:.2f}".format(r2_poly200, mse_poly200))