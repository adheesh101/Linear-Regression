import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("studytime_score_data.csv")

# Check column names
print(data.columns)

plt.scatter(data.studytime, data.score)
plt.show()

# Loss function
def loss_functions(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i]["studytime"]
        y = points.iloc[i]["score"]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))  # Fixed return statement

# Gradient Descent
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i]["studytime"]
        y = points.iloc[i]["score"]

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

# Initialize parameters
m = 0
b = 0
L = 0.0001
epochs = 10000

# Training loop
for i in range(epochs):
    if i % 100 == 0:
        loss = loss_functions(m, b, data)
        print(f"Epoch: {i}, Loss: {loss}")  # Print loss
    m, b = gradient_descent(m, b, data, L)

# Print final values
print("Final values of m and b:", m, b)

# Plot results
plt.scatter(data["studytime"], data["score"], color="black")
plt.plot(
    data["studytime"],
    [m * x + b for x in data["studytime"]],
    color="red"
)
plt.show()
