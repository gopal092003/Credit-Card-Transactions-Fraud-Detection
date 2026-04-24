from sklearn.model_selection import train_test_split

from src.data.load_data import load_data
from src.data.preprocess import preprocess_fraud_data
from src.features.build_features import add_time_features, add_distance_feature
from src.models.train_catboost import train_catboost
from src.models.train_xgboost import train_xgboost
from src.models.evaluate import evaluate_model
from src.utils.helpers import save_model, save_metrics, log


def run_training_pipeline(config):
    log("Loading data...")
    train, test = load_data(
        config["data"]["train_path"],
        config["data"]["test_path"],
    )

    log("Feature engineering...")
    train = add_time_features(train)
    test = add_time_features(test)

    train = add_distance_feature(train)
    test = add_distance_feature(test)

    log("Preprocessing...")
    train = preprocess_fraud_data(train)
    test = preprocess_fraud_data(test)

    target = config["features"]["target"]

    X = train.drop(columns=[target])
    y = train[target]

    log("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        stratify=y if config["training"]["stratify"] else None,
        random_state=config["training"]["random_state"],
    )

    model_type = config["model"]["type"]

    log(f"Training model: {model_type}")

    if model_type == "catboost":
        model = train_catboost(
            X_train,
            y_train,
            X_val,
            y_val,
            config["features"]["categorical"],
            config,
        )
        y_pred = model.predict(X_val)

    elif model_type == "xgboost":
        model = train_xgboost(
            X_train,
            y_train,
            config,
            config["features"]["low_cardinality"],
            config["features"]["high_cardinality"],
        )
        y_pred = model.predict(X_val)

    else:
        raise ValueError("Unsupported model type")

    log("Evaluating model...")
    metrics = evaluate_model(y_val, y_pred)

    print(metrics)

    if config["experiment"]["save_model"]:
        save_model(model, config["output"]["model_dir"], model_type)

    if config["experiment"]["save_metrics"]:
        save_metrics(metrics, config["output"]["metrics_dir"])

    return model, metrics