import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

def generate_data(d, num_points=2000):
    # Generate data points normally distributed around the origin
    X = np.random.normal(0, 1, size=(num_points, d))
    # Compute the target variable y
    y = X[:, 0] + np.sin(4 * X[:, 0])
    return X, y

def split_data(X, y):
    # Split into 1000 training and 1000 testing points
    return X[:1000], y[:1000], X[1000:], y[1000:]

def train_and_predict_knn(X_train, y_train, X_test, n_neighbors=5):
    # Train a 5-Nearest Neighbor Regressor
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # Predict on the test set
    y_pred = knn.predict(X_test)
    return y_pred

def compute_mse(y_true, y_pred):
    # Compute Mean Squared Error
    return np.mean((y_true - y_pred)**2)

def plot_results(x1_test, y_test, y_pred, d):
    plt.figure(figsize=(8, 6))
    plt.scatter(x1_test, y_test, color='blue', label='True Values', alpha=0.6, s=10)
    plt.scatter(x1_test, y_pred, color='red', label='Predicted Values', alpha=0.6, s=10)
    plt.title(f"Dimensionality: d = {d}")
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    dimensions = [1, 2, 3, 4, 7, 11, 16]
    mse_results = {}

    for d in dimensions:
        # Generate data
        X, y = generate_data(d)

        # Split into training and testing sets
        X_train, y_train, X_test, y_test = split_data(X, y)

        # Train and predict using 5-NN
        y_pred = train_and_predict_knn(X_train, y_train, X_test)

        # Compute mean squared error
        mse = compute_mse(y_test, y_pred)
        mse_results[d] = mse

        # Plot results
        plot_results(X_test[:, 0], y_test, y_pred, d)

    # Print MSE results
    print("Mean Squared Error for each dimensionality:")
    for d, mse in mse_results.items():
        print(f"Dimension {d}: MSE = {mse:.6f}")

if __name__ == "__main__":
    main()
