import numpy as np

from sklearn.datasets import load_iris, make_blobs
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def huberized_hinge_loss(y_true, y_pred, delta=1.0):
    z = y_true * y_pred
    loss = np.where(
        z >= 1,
        0,
        np.where(z >= 1 - delta, (1 - z) ** 2 / (2 * delta), 1 - z - (delta / 2)),
    )
    return np.mean(loss)


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.1, n_iters=1000) -> None:
        self.lr = learning_rate  # Learning rate
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iters = n_iters  # Number of iterations
        self.weights = None  # Weights
        self.bias = 0  # Bias
        self.classes = None
        self.models = None

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.models = {}

        for class_label in self.classes:
            y_k = np.where(y_train == class_label, 1, -1)
            self.weights = np.random.randn(X_train.shape[1]) * 0.01
            bias = 0

            # Update rule
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X_train):
                    pass
                    # condition = y_[index] * np.dot(self.weights, x_i) - self.bias >= 1
                    # if condition:
                    #     self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                    # else:
                    #     self.weights -= self.lr * (
                    #         2 * self.lambda_param * self.weights - np.dot(y_[index], x_i)
                    #     )
                    #     self.bias -= self.lr * y_[index]

        return self

    def predict(self, X_test):
        approx = np.dot(X_test, self.weights) - self.bias
        return np.sign(approx)

    def get_params(self, deep=True):
        return {
            "learning_rate": self.lr,
            "lambda_param": self.lambda_param,
            "n_iters": self.n_iters,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


if __name__ == "__main__":
    y_true = np.array([1, -1, 1, -1])
    y_pred = np.array([0.8, -0.5, 1.2, -1.5])
    loss = huberized_hinge_loss(y_true, y_pred, delta=1.0)
    print("Huberized Hinge Loss:", loss)
