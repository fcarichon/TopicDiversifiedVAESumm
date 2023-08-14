import sys
import re
import time
import json

import pandas as pd
import numpy as np
from scipy import stats

from tabulate import tabulate

INPUT_DIR = "../data/"
MAX_SIZE = 2100000

OUT_TRAIN_PATH = "20220510_amazon_reviews_train.csv"
OUT_VALID_PATH = "20220510_amazon_reviews_valid.csv"
OUT_TEST_PATH = "20220510_amazon_reviews_test.csv"


def load_categories():
    categories = []
    with open("./categories.txt", "r") as f:
        for line in f:
            categories.append(line.strip())
    return categories


def process(in_df, category, max_total_reviews, min_review_len, max_review_len):
    df = in_df.copy()

    # Add category information
    df["category"] = category

    # Step 1: Data Cleaning

    # Filter attributes
    df = df[["category", "asin", "reviewerID", "reviewTime", "reviewText", "overall"]]
    # Remove empty reviews
    df = df[~df["reviewText"].isnull()]
    # Remove duplicates
    df = df.drop_duplicates(subset=["reviewText", "asin", "reviewerID"], keep="first")
    # Remove reviews containing no alphabet letter
    df = df[df["reviewText"].apply(lambda x: re.search("[a-zA-Z]+", str(x)) is not None)]

    # Step 2: Filter by review length

    # Add review lengths information
    df["review_len"] = df["reviewText"].apply(lambda x: len(str(x).split()))
    # Filter data by review length
    df = df[(df["review_len"] >= min_review_len) & (df["review_len"] <= max_review_len)]

    return df


def split_dataset(data, num_test_prods=207, test_revs_per_prod=15, num_train_reviews=800, num_eval_reviews=200):
    # Sort categories in ascending order of number of reviews
    sorted_categories = data["category"].value_counts().index.values.tolist()

    # Assign categories to different sub datasets
    num_cat = len(sorted_categories)
    num_test_cat = int(max(1, np.round(num_cat * 0.14)))
    num_eval_cat = int(max(1, np.round(num_cat * 0.18)))
    # Sample the test data from the two largest categories as well as the two smallest.
    test_categories = sorted_categories[: int(np.ceil(num_test_cat / 2))] + sorted_categories[-int(np.floor(num_test_cat / 2)):]
    # Sample the valid and train data from the remaining categories
    valid_categories = sorted_categories[int(np.ceil(num_test_cat / 2)): int(np.ceil(num_test_cat / 2)) + num_eval_cat]
    train_categories = sorted_categories[int(np.ceil(num_test_cat / 2)) + num_eval_cat: -int(np.floor(num_test_cat / 2))]

    test_data = data[data["category"].apply(lambda x: str(x) in test_categories)].copy()
    eval_data = data[data["category"].apply(lambda x: str(x) in valid_categories)].copy()
    train_data = data[data["category"].apply(lambda x: str(x) in train_categories)].copy()

    def sample_test_data(in_df, num_prods, reviews_per_prod):
        prods = pd.Series(in_df["prod_id"].unique()).to_frame()
        prods = prods.sample(n=prods.shape[0], random_state=42)
        prods = prods[:num_prods].values.flatten().tolist()
        # In order to facilitate the manual labelling by human annotators, we limit the number of reviews per
        # product to a specific number
        df = None
        for prod in prods:
            reviews = in_df[in_df["prod_id"] == prod].copy()
            reviews = reviews.sample(n=min(reviews_per_prod, reviews.shape[0]), random_state=42)
            reviews = reviews.reset_index(drop=True)
            reviews["review_id"] = reviews.index.values
            df = pd.concat([df, reviews])
        return df

    def sample_data(df, num_reviews_per_cat):
        tmp = df.copy()
        df = None
        categories = tmp["category"].unique()
        for cat in categories:
            prods = pd.Series(tmp.loc[tmp["category"] == cat, "prod_id"].unique()).to_frame()
            prods = prods.sample(n=prods.shape[0], random_state=42)
            prods = prods[:].values.flatten().tolist()
            num_reviews = 0
            for prod in prods:
                reviews = tmp[tmp["prod_id"] == prod].copy()
                reviews = reviews.reset_index(drop=True)
                reviews["review_id"] = reviews.index.values
                df = pd.concat([df, reviews])
                num_reviews += reviews.shape[0]
                if num_reviews >= num_reviews_per_cat:
                    break
        return df

    # Sample data for the test set
    test_data = sample_test_data(test_data, num_test_prods, test_revs_per_prod)

    # Sample data for evaluation set
    eval_data = sample_data(eval_data, num_eval_reviews)

    # Sample data for train set
    train_data = sample_data(train_data, num_train_reviews)

    return train_data, eval_data, test_data


def build_dataset(max_num_reviews=350, max_total_reviews=100000, min_review_len=8, max_review_len=205,
                  min_num_reviews=15):
    start_time = time.perf_counter()

    dataset = None

    # Load list of categories
    print(f"Loading categories")
    categories = load_categories()

    for cat in categories:
        st = time.perf_counter()
        cat = "_".join(cat.split(" "))
        # Load data from file
        file_path = f"{INPUT_DIR}{cat}.json"
        print(f"Processing reviews from {file_path}")

        data = dict()
        with open(file_path, 'r') as file:
            for idx, line in enumerate(file):
                data[idx] = json.loads(line.strip())
                if idx > MAX_SIZE:
                    break
            print(f"Loaded {cat}. memory size: {(float(sys.getsizeof(data)) / 1000000.0):.2f} MB.")

            data = pd.DataFrame.from_dict(data, orient="index")

            # Process data for category
            df = process(data, cat, max_total_reviews, min_review_len, max_review_len)

            if dataset is None:
                dataset = df
            else:
                dataset = pd.concat([dataset, df])

            print(f"data memory size: {(float(sys.getsizeof(df)) / 1000000.0):.2f} MB. " +
                  f"time: {(time.perf_counter() - st):.2f} seconds")

    print(f"Finishing up processing...")

    # Step 6: Filter by number of reviews per product (remove 95th percentile)

    # Compute number of reviews per product
    num_revs_per_prod = dataset["asin"].value_counts().rename_axis('asin').reset_index(name='num_revs_per_prod')
    # max_num_reviews = num_revs_per_prod["num_revs_per_prod"].quantile([0.025, 0.975]).values.tolist()[1]
    max_num_reviews = min(max_num_reviews, num_revs_per_prod["num_revs_per_prod"].max())

    # Filter based on the number of reviews per product
    dataset = pd.merge(dataset, num_revs_per_prod, on="asin")
    dataset = dataset[
        (dataset["num_revs_per_prod"] >= min_num_reviews) & (dataset["num_revs_per_prod"] <= max_num_reviews)
        ]

    # Step 7: Clean up attributes

    keep_columns = ["category", "asin", "reviewText", "review_len"]
    dataset = dataset[keep_columns]
    # Rename columns
    dataset = dataset.rename(columns={
        "overall": "rating",
        "asin": "prod_id",
        "reviewText": "review"
    })

    dataset = dataset.reset_index(drop=True)

    # Step 8: Split datasets
    print(f"Splitting dataset...")
    train_data, eval_data, test_data = split_dataset(dataset)

    # Save datasets to file
    print("Saving datasets...")
    train_data.to_csv(OUT_TRAIN_PATH, index=False)
    eval_data.to_csv(OUT_VALID_PATH, index=False)
    test_data.to_csv(OUT_TEST_PATH, index=False)

    print(f"Completed in {(time.perf_counter() - start_time):.3f} seconds")

    statistics = tabulate(
        [
            [
                "Train", train_data.shape[0],
                train_data['category'].unique().shape[0],
                train_data['category'].value_counts().mean(),
                train_data['prod_id'].unique().shape[0],
                train_data['prod_id'].value_counts().mean()
            ],
            [
                "Eval", eval_data.shape[0],
                eval_data['category'].unique().shape[0],
                eval_data['category'].value_counts().mean(),
                eval_data['prod_id'].unique().shape[0],
                eval_data['prod_id'].value_counts().mean()
            ],
            [
                "Test", test_data.shape[0],
                test_data['category'].unique().shape[0],
                test_data['category'].value_counts().mean(),
                test_data['prod_id'].unique().shape[0],
                test_data['prod_id'].value_counts().mean()
            ]
        ],
        headers=["dataset", "size", "num categories", "avg num revs per cat", "num products", "avg num revs per prod"]
    )

    print()
    print(statistics)

    return dataset


if __name__ == "__main__":
    build_dataset()
