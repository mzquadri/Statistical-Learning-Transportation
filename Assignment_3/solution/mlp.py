"""Problem Set 3, Problem 1 - Mohd Zamin Quadri

Completed version of the MLP skeleton supplied with the problem set.

Changes to the supplied file:
  * the five ### YOUR CODE ### gaps are filled in;
  * `alpha` is added as a constructor argument.  The supplied skeleton defines no
    training hyperparameter of that name (its only `alpha` is the plot-transparency
    argument in the test snippet), so Problem 1.2 requires an interpretation; I
    assume it is the coefficient of an L2 penalty on the weights, as in sklearn's
    MLPClassifier.  It defaults to 0.0, so the unregularised behaviour of the
    original skeleton is recovered exactly;
  * the output is clipped inside the loss and its derivative to keep the computation
    finite near probabilities 0 and 1.  No learning rate in the Problem 1.1 sweep
    produced a non-finite loss;
  * the test snippet seeds numpy, because init_parameters draws from np.random.
"""

import numpy as np

# Keeps log(out) and 1/(out(1-out)) finite when a large learning rate saturates
# the output sigmoid.  float64 resolves 1 - 1e-12 exactly, so this is a safe bound.
EPS = 1e-12


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01,
                 alpha=0.0):
        # Set hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.alpha = alpha  # L2 coefficient; 0.0 reproduces the original skeleton

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
        # backward_propagation passes the *activation output* z = sigma(a), not the
        # pre-activation. Written in terms of its own output, sigma' = z (1 - z).
        return a * (1 - a)
        ### YOUR CODE ENDS HERE ###

    def forward_propagation(self, X):
        self.a1 = X @ self.W1 + self.b1
        self.z1 = self.activation(self.a1)
        ### YOUR CODE STARTS HERE ###
        self.a2 = self.z1 @ self.W2 + self.b2
        self.out = self.activation(self.a2)
        return self.out
        ### YOUR CODE ENDS HERE ###

    def binary_cross_entropy_loss(self, y, out):
        out = np.clip(out, EPS, 1 - EPS)
        return -np.mean(y * np.log(out) + (1 - y) * np.log(1 - out))

    def binary_cross_entropy_derivative(self, y, out):
        ### YOUR CODE STARTS HERE ###
        # d/dout of the *mean* loss, so the 1/N of np.mean has to be carried along.
        # The clip guards this division only.  backward_propagation then multiplies
        # the result by activation_derivative(out) using the *unclipped* out, which
        # is the skeleton's own line and is left unaltered; were the sigmoid ever to
        # saturate to exactly 0.0 or 1.0, that product would be 0 rather than the
        # analytic (out - y)/N.
        out = np.clip(out, EPS, 1 - EPS)
        return (out - y) / (y.size * out * (1 - out))
        ### YOUR CODE ENDS HERE ###

    def regularisation_term(self):
        """L2 penalty (alpha/2)(||W1||^2 + ||W2||^2); biases are not penalised."""
        return 0.5 * self.alpha * ((self.W1 ** 2).sum() + (self.W2 ** 2).sum())

    def total_loss(self, y, out):
        """Objective actually minimised: data term + L2 penalty."""
        return self.binary_cross_entropy_loss(y, out) + self.regularisation_term()

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
        # b was broadcast over the N rows in the forward pass, so its gradient is the
        # column sum.  alpha * W is the derivative of the L2 penalty; biases are exempt.
        grad_W2 = self.z1.T @ grad_a2 + self.alpha * self.W2
        grad_b2 = grad_a2.sum(axis=0, keepdims=True)
        grad_W1 = X.T @ grad_a1 + self.alpha * self.W1
        grad_b1 = grad_a1.sum(axis=0, keepdims=True)
        ### YOUR CODE ENDS HERE ###

        self.update_weights(grad_W2, grad_b2, grad_W1, grad_b1)

    def update_weights(self, grad_W2, grad_b2, grad_W1, grad_b1):
        # Update weights and biases using gradient descent
        ### YOUR CODE STARTS HERE ###
        self.W2 = self.W2 - self.learning_rate * grad_W2
        self.b2 = self.b2 - self.learning_rate * grad_b2
        self.W1 = self.W1 - self.learning_rate * grad_W1
        self.b1 = self.b1 - self.learning_rate * grad_b1
        ### YOUR CODE ENDS HERE ###

    def fit(self, X, y, epochs=10000, verbose=True):
        # Train the model for some epochs
        history = []
        for epoch in range(epochs):
            output = self.forward_propagation(X)
            history.append(self.total_loss(y, output))
            if not np.isfinite(history[-1]):
                # A learning rate past the stability limit sends the weights to inf,
                # after which every later epoch is nan. Stop and let the caller see it.
                break
            self.backward_propagation(X, y, output)
            if verbose and epoch % 1000 == 0:
                loss = self.binary_cross_entropy_loss(y, output)
                print(f"Epoch {epoch}, Loss: {loss}")
        return np.array(history)

    def predict(self, X):
        # Get the predicted classes
        output = self.forward_propagation(X)
        return np.round(output)

    def test(self, X, y):
        # Evaluate the model on test data
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Accuracy: {accuracy * 100}%")
        return accuracy


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
    np.random.seed(42)  # added: init_parameters draws from np.random
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
