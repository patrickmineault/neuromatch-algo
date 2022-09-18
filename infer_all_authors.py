import lib
from tqdm import tqdm
import pandas as pd


def main():
    for conference in ["nmc1", "nmc2", "nmc3"]:
        df_users = pd.read_json(f"data/processed/{conference}.json", orient="table")
        author_ids = []
        for _, row in tqdm(df_users.iterrows()):
            author_id = lib.infer_author_id(row)
            author_ids.append(author_id)
            print(author_id)

        df_users["author_id"] = author_ids

        pd.to_json(f"data/processed/{conference}-with-authors.json", orient="table")


if __name__ == "__main__":
    main()
