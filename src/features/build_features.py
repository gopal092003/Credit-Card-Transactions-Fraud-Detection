import numpy as np


def add_time_features(df):
    """
    Create time-based features
    """
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).copy()

    df["month"] = df["trans_date_trans_time"].dt.month
    df["day"] = df["trans_date_trans_time"].dt.day

    def get_part_of_day(hour):
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    df["part_of_day"] = df["trans_date_trans_time"].dt.hour.apply(get_part_of_day)

    df["time_since_last"] = (
        df.groupby("cc_num")["trans_date_trans_time"]
        .diff()
        .dt.total_seconds()
    ).fillna(0)

    return df


def add_distance_feature(df):
    """
    Compute Haversine distance between customer and merchant
    """
    coord_cols = ["lat", "long", "merch_lat", "merch_long"]

    if not all(col in df.columns for col in coord_cols):
        return df

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    df["distance"] = haversine(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    return df