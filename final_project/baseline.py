from sklearn.ensemble import RandomForestClassifier

class BaselineModel:
    def __init__(self, n_estimators=100, random_state=42):
        """
        Args:
            n_estimators: Number of trees in the forest.
            random_state: Random seed for reproducibility.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X_train, y_train):
        """
        Train the RandomForestClassifier.

        Args:
            X_train: Training features.
            y_train: Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict class labels for the test set.

        Args:
            X_test: Test features.

        Returns:
            Predicted class labels.
        """
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Predict class probabilities for the test set.

        Args:
            X_test: Test features.

        Returns:
            Predicted class probabilities.
        """
        return self.model.predict_proba(X_test)