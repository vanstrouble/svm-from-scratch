import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.1, n_iters=1000) -> None:
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0

    def fit(self, X_train, y_train):
        y_ = np.where(y_train <= 0, -1, 1)
        self.weights = np.random.randn(X_train.shape[1]) * 0.01

        # Update rule
        for _ in range(self.n_iters):
            for index, x_i in enumerate(X_train):
                condition = y_[index] * np.dot(self.weights, x_i) - self.bias >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (
                        2 * self.lambda_param * self.weights - np.dot(y_[index], x_i)
                    )
                    self.bias -= self.lr * y_[index]

    def predict(self, X_test):
        approx = np.dot(X_test, self.weights) - self.bias
        pass


if __name__ == "__main__":
    pass
