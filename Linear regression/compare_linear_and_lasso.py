from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

#Generate artificial data for regression lines. 
x = np.random.normal(0, 10, (20, 1))
y = 4*x + 10 + np.random.normal(0, 5, (20, 1))

regr = linear_model.LinearRegression()
LASSO = linear_model.Lasso(alpha=50) #50 is a large value of alpha, and is chosen for demonstration purposes. 
plot_x = np.transpose(np.array([list(range(-20, 20))]))
regr.fit(x, y)
LASSO.fit(x, y)

plt.plot(plot_x, regr.predict(plot_x))
plt.plot(plot_x, LASSO.predict(plot_x))
plt.legend(["Linear Regression", "LASSO"])
plt.plot(x, y, "ro")
plt.axis([-15, 15, -70, 70])
plt.savefig("Plots.png", format="png")
