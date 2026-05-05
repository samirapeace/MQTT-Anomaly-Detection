import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, label_column="label"):
        self.label_column = label_column

    def clean_labels(self, df):
        if self.label_column in df.columns:
            df[self.label_column] = (
                df[self.label_column]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(" ", "")
            )
        return df

    def remove_duplicates(self, df):
        return df.drop_duplicates()

  
    def handle_missing(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns


        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())


        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")

        return df


    def remove_constant_columns(self, df):
        nunique = df.nunique()
        constant_cols = nunique[nunique <= 1].index
        return df.drop(columns=constant_cols, errors="ignore")


    def remove_text_columns(self, df):
        text_cols = [
            "mqtt.msg",
            "mqtt.willmsg",
            "mqtt.protoname",
            "mqtt.willtopic"
        ]
        return df.drop(columns=text_cols, errors="ignore")


    def convert_to_numeric(self, df):
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = df[col].astype("category").cat.codes
                except:
                    pass
        return df


    def clip_outliers(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(q1, q99)

        return df


    def clean_chunk(self, df):
        df = self.clean_labels(df)
        df = self.remove_duplicates(df)
        df = self.remove_text_columns(df)
        df = self.handle_missing(df)
        df = self.convert_to_numeric(df)
        df = self.remove_constant_columns(df)
        df = self.clip_outliers(df)

        return df


def class_distribution(df, label_column="label"):
    return df[label_column].value_counts()
