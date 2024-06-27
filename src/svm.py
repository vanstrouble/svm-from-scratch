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

        return self

    def predict(self, X_test):
        approx = np.dot(X_test, self.weights) - self.bias
        return np.sign(approx)

    def get_params(self, deep=True):
        return {
            "learning_rate": self.lr,
            "lambda_param": self.lambda_param,
            "n_iters": self.n_iters
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


if __name__ == "__main__":
    from sklearn.datasets import load_iris, make_blobs
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score

    X, y = load_iris(return_X_y=True)
    X = X[:100, [0, 2]]
    y = y[:100]
    y = np.where(y == 0, -1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=123
    )

    param_grid = {
        "learning_rate": [0.0001, 0.001, 0.01, 0.1],
        "lambda_param": [0.001, 0.01, 0.1, 1],
        "n_iters": [100, 1000, 5000, 10000],
    }

    grid_search = GridSearchCV(SVM(), param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    best_svm, best_params = grid_search.best_estimator_, grid_search.best_params_
    y_pred = best_svm.predict(X_test)

    # svm = SVM()
    # svm.fit(X_train, y_train)

    # y_pred = svm.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    # X, y = make_blobs(
    #     n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=123
    # )
    # y = np.where(y == 0, -1, 1)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=123
    # )

    # svm = SVM()
    # svm.fit(X_train, y_train)
    # y_pred = svm.predict(X_test)

    # print(accuracy_score(y_test, y_pred))
