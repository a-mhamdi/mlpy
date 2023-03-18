import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 0], [1, 25], [1, 50], [1, 75], [1, 100]])
y = np.array([14, 38, 54, 76, 95])

# NORMAL EQUATION
theta_ne = np.linalg.inv(X.T @ X) @ X.T @ y

# GRADIENT DESCENT
theta_gd = np.zeros(shape=(2, 1001))
theta_gd[:, 0] = np.array([10, .5])
cost = []
for k in range(1000):
    eps = y-(X @ theta_gd[:, k])
    cost.append(1/5*(eps @ eps))
    theta_gd[:, k+1] = theta_gd[:, k] + .003/5*(eps @ X)

plt.plot(theta_gd[0, :], label=r'$\hat{\theta}_0$')
plt.plot(theta_gd[1, :], label=r'$\hat{\theta}_1$')
plt.legend()
plt.grid()
plt.show()

plt.plot(cost)
plt.grid()
plt.show()
