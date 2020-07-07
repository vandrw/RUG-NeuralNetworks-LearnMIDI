import matplotlib.pyplot as plt
from numpy import genfromtxt

plt.rc('axes', labelsize=14) 

cross_valid = genfromtxt('gan_model/losses.csv', delimiter=',')
generator = cross_valid[:, 0]
discriminator = cross_valid[:, 1]

plt.plot(generator, label="Generator")
plt.plot(discriminator, label="Discriminator")
plt.xlabel("Epoch")
plt.ylabel("Loss (Binary Crossentropy)")
plt.legend()
plt.savefig("gan_model/losses.png")