'''from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
model = LinearRegression()
x = np.arange(5000,85000,10000).reshape(8,1)
y = np.array([22,20,18,16,15,13,12,10])
model.fit(x,y)
print(model.predict([[60000]]))
print(model.coef_)
print(model.intercept_)
# Predict values for the original x range
y_pred = model.predict(x)

# Plot actual data points
plt.scatter(x, y, color='blue', label='Actual Data')

# Plot regression line
plt.plot(x, y_pred, color='red', label='Regression Line')

plt.xlabel('Mileage (km)')
plt.ylabel('Price ($1000s)')
plt.title('Car Price vs. Mileage')
plt.legend()
plt.grid(True)
plt.show()'''


'''from sklearn.linear_model import LinearRegression
import numpy as np
model = LinearRegression()
x = np.array([[4,128],
              [8,256],
              [8,512],
              [16,256],
              [16,512],
              [32,512],
              [32,1024],
              [64,1024]])
y = np.array([0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0])
model.fit(x,y)
print(model.predict([[16,1024]]))
print(model.coef_[0])
print(model.intercept_)'''

'''from sklearn.linear_model import LinearRegression
import numpy as np
model = LinearRegression()
x = np.array([
    [4, 64, 12],
    [6, 128, 12],
    [8, 128, 16],
    [8, 256, 48],
    [12, 256, 48],
    [12, 512, 64],
    [16, 512, 108],
    [16, 1024, 200]
])
y = np.array([0.6, 0.8, 1.0, 1.3, 1.5, 1.8, 2.2, 2.8])
model.fit(x,y)
print(model.predict([[12,1024,108]]))
print(model.coef_)
print(model.intercept_)'''

from sklearn.linear_model import LinearRegression
import numpy as np
model = LinearRegression()
x = np.array([
    [165000, 130000, 120000],
    [150000, 140000, 115000],
    [130000, 125000, 100000],
    [120000, 115000, 105000],
    [100000, 100000, 95000],
    [95000,  90000,  97000],
    [85000,  85000, 91000],
    [80000,  70000, 88000]
])
y = np.array([192, 185, 170, 160, 150, 145, 130, 125])
model.fit(x,y)
print(model.predict([[110000,90000,100000]]))




