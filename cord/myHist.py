import numpy as np
import matplotlib.pyplot as plt
import sys

file = open(sys.argv[1])
data = file.readlines()
train_d = [float(i) for i in data[0].split(",")]
train_t = [float(i) for i in data[1].split(",")]
test_d = [float(i) for i in data[2].split(",")]
test_t = [float(i) for i in data[3].split(",")]

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.hist(train_d, bins=10, alpha=0.5, label="x")
ax1.hist(train_t, bins=10, alpha=0.5, label="y")
ax1.set_title("train")
plt.xlabel('bins')
plt.ylabel('output')

ax1 = fig.add_subplot(1,2,2)
ax1.hist(test_d, bins=10, alpha=0.5, label="x")
ax1.hist(test_t, bins=10, alpha=0.5, label="y")
ax1.set_title("test")
plt.xlabel('bins')
plt.ylabel('output')

plt.legend()
plt.show()
