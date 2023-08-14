import configs.config as config
from data.preprocess import TextProcessing
import re

import pandas as pd
import time
from tqdm import tqdm

DATA_DIR = config.path["data"]
#FILE_PATH = f"{DATA_DIR}amazon_reviews_2018.csv"


def preprocess(file_name):
    
    processing = TextProcessing()
    
    tqdm.pandas()
    FILE_PATH = f"{DATA_DIR}{file_name}"
    
    df = pd.read_csv(FILE_PATH)
    st = time.perf_counter()
    print(f"Processing data at '{FILE_PATH}'")
    #df["review"] = df["review"].apply(processing.preprocess)
    df["review"] = df["review"].progress_apply(processing.preprocess)
    
    new_path = re.sub('.csv', '', file_name)
    new_path = new_path + '_processed.csv'
    df.to_csv(f"{DATA_DIR}{new_path}", index=False)
    print(f"{new_path} Completed after {(time.perf_counter() - st):.2f} seconds")


if __name__ == "__main__":
    preprocess()