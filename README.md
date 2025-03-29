# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load your dataset
file_path = "/content/AirPassengers.csv"
df = pd.read_csv(file_path, parse_dates=['Month'], index_col='Month')

# Resample data yearly and convert index to year
resampled_data = df['#Passengers'].resample('Y').sum().to_frame()
resampled_data.index = resampled_data.index.year
resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'Month': 'Year'}, inplace=True)

# Extract values for trend estimation
years = resampled_data['Year'].tolist()
passengers = resampled_data['#Passengers'].tolist()

# Linear Trend Estimation
X = [i - years[len(years) // 2] for i in years]  # Centering years around the middle
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, passengers)]
n = len(years)

# Solve for linear trend coefficients
b = (n * sum(xy) - sum(passengers) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(passengers) - b * sum(X)) / n

# Generate linear trend values
linear_trend = [a + b * X[i] for i in range(n)]

# Polynomial Trend Estimation (Degree 2)
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, passengers)]

# Construct matrix for polynomial regression
coeff = [[n, sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]
Y = [sum(passengers), sum(xy), sum(x2y)]

# Solve for polynomial coefficients
A = np.array(coeff)
B = np.array(Y)
solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution

# Generate polynomial trend values
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

# Display equations
print(f"Linear Trend: y = {a:.2f} + {b:.2f}x")
print(f"Polynomial Trend: y = {a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

# Add trends to DataFrame
resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend
resampled_data.set_index('Year', inplace=True)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(resampled_data.index, resampled_data['#Passengers'], 'bo-', label='Actual Data')
plt.plot(resampled_data.index, resampled_data['Linear Trend'], 'k--', label='Linear Trend')
plt.plot(resampled_data.index, resampled_data['Polynomial Trend'], 'go-', label='Polynomial Trend')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.title('Trend Analysis')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT:

![image](https://github.com/user-attachments/assets/3f3582ee-00e9-4b4e-aea8-07cd7c97ad26)


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
