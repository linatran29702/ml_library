from sklearn.tree import DecisionTreeRegressor

class TreeModel:
    def __init__(self, max_depth=None):
        self.model = DecisionTreeRegressor(max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
