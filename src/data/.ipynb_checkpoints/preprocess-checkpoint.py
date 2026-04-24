def preprocess_fraud_data(df):
    """
    Clean and preprocess dataset
    """
    df = df.copy()

    drop_cols = [
        "Unnamed: 0",
        "cc_num",
        "first",
        "last",
        "street",
        "city",
        "trans_num",
    ]

    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Age feature
    if "dob" in df.columns and "trans_date_trans_time" in df.columns:
        df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
        df = df.drop(columns=["dob"])

    # Drop datetime after extraction
    if "trans_date_trans_time" in df.columns:
        df = df.drop(columns=["trans_date_trans_time"])

    # Drop raw coordinates AFTER feature creation
    coord_cols = ["lat", "long", "merch_lat", "merch_long"]
    df = df.drop(columns=[col for col in coord_cols if col in df.columns])

    return df