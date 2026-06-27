import numpy as np
import matplotlib.pyplot as plt


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

def train_perceptron(X, y, lr=0.1, epochs=10):

    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):

        for i in range(len(X)):

           
            linear_output = np.dot(X[i], weights) + bias

           
            y_pred = 1 if linear_output >= 0 else 0

         
            update = lr * (y[i] - y_pred)

            weights += update * X[i]
            bias += update

    return weights, bias

weights, bias = train_perceptron(X, y)

print("Final Weights:", weights)
print("Final Bias:", bias)

print("\nPredictions")
for i in range(len(X)):
    output = np.dot(X[i], weights) + bias
    pred = 1 if output >= 0 else 0
    print(f"{X[i]} --> {pred}")

plt.figure(figsize=(6,6))

for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1],
                    color='red',
                    s=120,
                    marker='o',
                    label='Class 0' if i==0 else "")
    else:
        plt.scatter(X[i][0], X[i][1],
                    color='blue',
                    s=120,
                    marker='^',
                    label='Class 1')

x_values = np.linspace(-0.2, 1.2, 100)

if weights[1] != 0:
    y_values = -(weights[0]*x_values + bias)/weights[1]
    plt.plot(x_values, y_values,
             color='green',
             linewidth=2,
             label='Decision Boundary')

plt.xlim(-0.2,1.2)
plt.ylim(-0.2,1.2)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron Decision Boundary (AND Gate)")
plt.grid(True)
plt.legend()

plt.show()