from catboost import CatBoostClassifier


def train_catboost(X_train, y_train, X_val, y_val, cat_cols, config):
    params = config["model"]["catboost"]

    model = CatBoostClassifier(
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        loss_function=params["loss_function"],
        eval_metric=params["eval_metric"],
        scale_pos_weight=params["scale_pos_weight"],
        random_seed=params["random_seed"],
        verbose=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    return model