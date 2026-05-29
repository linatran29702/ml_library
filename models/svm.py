from sklearn.svm import SVR

class SVMModel:
    def __init__(self, kernel="rbf", C=1.0):
        self.model = SVR(kernel=kernel, C=C)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
