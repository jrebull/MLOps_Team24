from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

class ModelTrainer:
    def __init__(self, df: pd.DataFrame, target_col: str = "Class"):
        self.df = df
        self.target_col = target_col
        self.model = None

    def split_and_scale(self, test_size: float = 0.2, random_state: int = 42):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def train_logistic(self):
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self.X_train, self.y_train)

    def get_trained_model(self):
        return self.model
