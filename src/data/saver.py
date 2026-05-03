import os
import pandas as pd

class DataSaver:
    def __init__(self, path):
        self.path = path

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    # =========================
    # Safe Save 
    # =========================
    def save_chunk(self, df, first=False):

        if df is None or df.empty:
            print("⚠️ Empty chunk, skipping...")
            return

        try:
            if first:
                df.to_csv(self.path, mode='w', index=False)
            else:
                df.to_csv(self.path, mode='a', index=False, header=False)

        except Exception as e:
            print(f"❌ Error saving chunk: {e}")
