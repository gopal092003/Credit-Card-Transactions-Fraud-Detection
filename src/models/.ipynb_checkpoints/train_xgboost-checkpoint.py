from xgboost import XGBClassifier
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def train_xgboost(X_train, y_train, config, low_card, high_card):
    params = config["model"]["xgboost"]

    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    preprocess = ColumnTransformer(
        transformers=[
            ("target_enc", TargetEncoder(), high_card),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card),
        ],
        remainder="passthrough",
    )

    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        eval_metric=params["eval_metric"],
        scale_pos_weight=scale_pos_weight,
        random_state=params["random_state"],
    )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    return pipeline