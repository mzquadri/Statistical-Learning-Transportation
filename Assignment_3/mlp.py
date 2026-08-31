import numpy as np


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Set hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.init_parameters()

    def init_parameters(self):
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def activation(self, a):
        denom = 1 + np.exp(-a)
        return 1 / denom

    def activation_derivative(self, a):
        ### YOUR CODE STARTS HERE ###
        return ...
        ### YOUR CODE ENDS HERE ###

    def forward_propagation(self, X):
        self.a1 = X @ self.W1 + self.b1
        self.z1 = self.activation(self.a1)
        ### YOUR CODE STARTS HERE ###
        return ...
        ### YOUR CODE ENDS HERE ###

    def binary_cross_entropy_loss(self, y, out):
        return -np.mean(y * np.log(out) + (1 - y) * np.log(1 - out))

    def binary_cross_entropy_derivative(self, y, out):
        ### YOUR CODE STARTS HERE ###
        return ...
        ### YOUR CODE ENDS HERE ###

    def backward_propagation(self, X, y, out):
        # ∇L_out
        grad_out = self.binary_cross_entropy_derivative(y, out)

        # ∇L_a2
        grad_a2 = grad_out * self.activation_derivative(out)

        # ∇L_z1
        grad_z1 = grad_a2 @ self.W2.T

        # ∇L_a1
        grad_a1 = grad_z1 * self.activation_derivative(self.z1)

        # Gradient of weights and biases
        ### YOUR CODE STARTS HERE ###
        grad_W2 = ...
        grad_b2 = ...
        grad_W1 = ...
        grad_b1 = ...
        ### YOUR CODE ENDS HERE ###

        self.update_weights(grad_W2, grad_b2, grad_W1, grad_b1)

    def update_weights(self, grad_W2, grad_b2, grad_W1, grad_b1):
        # Update weights and biases using gradient descent
        ### YOUR CODE STARTS HERE ###
        self.W2 = ...
        self.b2 = ...
        self.W1 = ...
        self.b1 = ...
        ### YOUR CODE ENDS HERE ###

    def fit(self, X, y, epochs=10000):
        # Train the model for some epochs
        for epoch in range(epochs):
            output = self.forward_propagation(X)
            self.backward_propagation(X, y, output)
            if epoch % 1000 == 0:
                loss = self.binary_cross_entropy_loss(y, output)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        # Get the predicted classes
        output = self.forward_propagation(X)
        return np.round(output)

    def test(self, X, y):
        # Evaluate the model on test data
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Accuracy: {accuracy * 100}%")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_circles
    from sklearn.model_selection import train_test_split

    # Generate a dataset
    X, y = make_circles(200, noise=0.1, random_state=42)
    y = y[:, np.newaxis]
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Initialize and train the MLP
    mlp = MLP(input_size=2, hidden_size=8, output_size=1, learning_rate=0.001)
    mlp.fit(X_train, y_train, epochs=20000)

    # Test the MLP
    mlp.test(X_test, y_test)

    plt.figure(figsize=(5, 5))
    # Scatter plot of the data points
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), ec="black", cmap="coolwarm")

    # Decision surface of the model
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    out = mlp.forward_propagation(X_grid)
    plt.pcolormesh(xx, yy, out.reshape(xx.shape), cmap="coolwarm", alpha=0.5)
    plt.show()
