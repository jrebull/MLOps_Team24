import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def encode_labels(self, target_col: str = "Class"):
        if target_col in self.df.columns:
            self.df[target_col] = self.encoder.fit_transform(self.df[target_col])

    def drop_duplicates(self):
        self.df.drop_duplicates(inplace=True)

    def drop_constant_columns(self):
        const_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        self.df.drop(columns=const_cols, inplace=True)

    def get_clean_df(self) -> pd.DataFrame:
        return self.df