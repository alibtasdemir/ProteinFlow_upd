import os, glob
import pandas as pd


def update_path(path):
    name = os.path.basename(path)
    base_folder = "preprocessed"
    return os.path.join(base_folder, name)


df = pd.read_csv("preprocessed/metadata.csv")
df["processed_path"] = df["processed_path"].apply(update_path)
df.to_csv("preprocessed/metadata_desktop.csv", index=None)
