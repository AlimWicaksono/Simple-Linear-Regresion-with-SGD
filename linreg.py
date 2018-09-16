import numpy as np
import matplotlib.pyplot as plt

#Artificial dataset
np.random.seed(11235)
x_data = np.arange(0., 16., 0.2)
np.random.shuffle(x_data)
y_data = np.array(x_data * 2.85) + np.random.normal(scale=2.5, size=x_data.shape)

w = 0.0
lr = 0.01

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

def gradient(x, y):
    return 2 * x * (x * w - y)

loss_list = []
n_epochs = 10
batch_size = 20
n_iter = len(x_data) // batch_size

for epoch in range(n_epochs):
    for i in range(n_iter - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        grad = loss_val = 0.
        for x_val, y_val in zip(x_data[start:end], y_data[start:end]):
            grad += gradient(x_val, y_val)
            loss_val += loss(x_val, y_val)
        w = w - lr * grad / batch_size
        loss_list.append(loss_val)

plt.plot(range(len(loss_list)), loss_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

plt.plot(x_data, y_data, 'go',  markersize=1.5)
plt.plot(x_data, forward(x_data), linewidth=.5)
plt.ylabel('y')
plt.ylabel('y')
plt.xlabel('x')
plt.show()
