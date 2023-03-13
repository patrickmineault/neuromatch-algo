import numpy as np
import pandas as pd


def main():

    df_info = pd.read_csv("data/input/Cosyne2023-Mar07-info.csv")
    df_info.Email = df_info.Email.str.lower().str.strip()
    df_info.Email = np.where(df_info.AltEmail.isna(), df_info.Email, df_info.AltEmail)

    df_registrants = pd.read_csv("data/input/Cosyne2023-Mar07-registrants.csv")
    df_registrants.Email = df_registrants.Email.str.lower().str.strip()
    df_registrants = df_registrants[df_registrants.Email != "cancelled"]

    # Reject Megan Peters because she hosting the event.
    if (df_registrants.Email == "megan.peters@uci.edu").sum() > 0:
        print("Removing Megan Peters from the registrants as host of Cosyne")
    df_registrants = df_registrants[df_registrants.Email != "megan.peters@uci.edu"]

    if df_registrants.Email.duplicated().sum() > 0:
        print("Duplicate emails in the registrants")
        print(df_registrants.Email.value_counts().head())
        return

    users = df_registrants.merge(df_info, left_on="Email", right_on="Email", how="left")

    bad_emails = df_registrants[~df_registrants.Email.isin(users.Email.values)].Email
    if len(bad_emails) > 0:
        print("Bad emails in the airtable")
        print(bad_emails.values)

    users["fullname"] = users.NameFirst + " " + users.NameLast
    users["user_id"] = users["Email"]

    users.drop(
        columns=["NameFirst", "NameLast", "AltEmail", "Name", "Email"],
        inplace=True,
    )

    if users.user_id.duplicated().sum() > 0:
        print("Duplicate user id in the users")
        print(users.user_id.value_counts().head())
        return

    users = users[~users.user_id.duplicated(keep="last")]
    users = users.reset_index(drop=True)

    users["abstracts"] = (
        users.Abstract1.fillna("")
        + "\n\n"
        + users.Abstract2.fillna("")
        + "\n\n"
        + users.Abstract3.fillna("")
    )
    assert users["abstracts"].isna().sum() == 0
    users.abstracts = users.abstracts.str.strip()
    users.abstracts = np.where(users.abstracts == "", np.nan, users.abstracts)

    users.to_pickle("data/transformed/match_users_filled.pkl")


if __name__ == "__main__":
    main()
