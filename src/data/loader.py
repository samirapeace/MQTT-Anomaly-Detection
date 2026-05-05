from pathlib import Path
import pandas as pd

class DataLoader:
    def __init__(self, data_path, chunksize=100_000):
        self.data_path = Path(data_path)
        self.chunksize = chunksize
        self.csv_files = list(self.data_path.rglob("*.csv"))

    def get_files(self):
        return self.csv_files


    def load_in_chunks(self, file_path):
        try:
            return pd.read_csv(
                file_path,
                chunksize=self.chunksize,
                low_memory=False,
                encoding="utf-8",
                on_bad_lines="skip"
            )
        except:
            return pd.read_csv(
                file_path,
                chunksize=self.chunksize,
                low_memory=False,
                encoding="latin1",
                on_bad_lines="skip"
            )


    def dataset_generator(self):
        for file in self.csv_files:
            print(f"📂 Loading: {file}")

            try:
                for chunk in self.load_in_chunks(file):

                    chunk = chunk.dropna(axis=1, how="all")

                    chunk.columns = [col.strip() for col in chunk.columns]

                    yield chunk

            except Exception as e:
                print(f"❌ Error in {file}: {e}")
