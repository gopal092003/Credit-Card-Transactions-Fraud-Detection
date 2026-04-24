import pandas as pd


def load_data(train_path: str, test_path: str):
    """
    Load train and test datasets with date parsing
    """
    train = pd.read_csv(
        train_path,
        parse_dates=["trans_date_trans_time", "dob"],
        low_memory=False,
    )

    test = pd.read_csv(
        test_path,
        parse_dates=["trans_date_trans_time", "dob"],
        low_memory=False,
    )

    return train, test