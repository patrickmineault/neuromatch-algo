import lib
import os
import pandas as pd
from tqdm import tqdm


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with open("infer.log", "w") as f:
        for conference in ["nmc1", "nmc2", "nmc3"]:
            df_users = pd.read_json(f"data/processed/{conference}.json", orient="table")
            author_ids = []

            for _, row in tqdm(df_users.iterrows()):
                author_id = lib.infer_author_id(row)
                author_ids.append(author_id)
                f.write(f"{author_id}\n")

            df_users["author_id"] = author_ids

            df_users.to_json(
                f"data/processed/{conference}-with-authors.json", orient="table"
            )


if __name__ == "__main__":
    main()
