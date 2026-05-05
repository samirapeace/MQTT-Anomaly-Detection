import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import StandardScaler

class FeatureSelector:
    def __init__(self, label_column="label"):
        self.label_column = label_column


    def remove_correlated(self, df, threshold=0.9):
        corr_matrix = df.corr(numeric_only=True).abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper.columns
            if any(upper[column] > threshold)
        ]

        return df.drop(columns=to_drop), to_drop


    def remove_low_variance(self, df, threshold=0.01):
        X = df.drop(columns=[self.label_column])

        selector = VarianceThreshold(threshold=threshold)
        X_var = selector.fit_transform(X)

        selected_columns = X.columns[selector.get_support()]

        df_new = pd.DataFrame(X_var, columns=selected_columns)
        df_new[self.label_column] = df[self.label_column].values

        return df_new, selected_columns


    def select_by_mi(self, df, top_k=25):
        X = df.drop(columns=[self.label_column])
        y = df[self.label_column]


        y_encoded = y.astype("category").cat.codes

        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        
        mi = mutual_info_classif(X_scaled, y_encoded, random_state=42)

        mi_series = pd.Series(mi, index=X.columns)

        
        mi_series = mi_series[mi_series > 0.001]


        top_features = mi_series.sort_values(ascending=False).head(top_k)

        selected_columns = top_features.index

        return df[selected_columns.tolist() + [self.label_column]], selected_columns


    def full_selection(self, df, top_k=25):
        print("🔹 Removing low variance features...")
        df, _ = self.remove_low_variance(df)

        print("🔹 Removing correlated features...")
        df, _ = self.remove_correlated(df)

        print("🔹 Selecting best features with MI...")
        df, selected = self.select_by_mi(df, top_k=top_k)

        print(f"✅ Selected {len(selected)} features")

        return df, selected
