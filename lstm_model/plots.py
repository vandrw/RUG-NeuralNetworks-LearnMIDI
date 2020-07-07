import matplotlib.pyplot as plt
from numpy import genfromtxt

cross_valid1 = genfromtxt('cross-valid1.csv', delimiter=',')

plt.plot(cross_valid1[:, 0], label="Training loss")
plt.plot(cross_valid1[:, 1], label="Testing loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (Binary Crossentropy)")
plt.show()


