from src.utils.helpers import load_config, log
from src.pipelines.training_pipeline import run_training_pipeline


def main():
    log("Starting training pipeline...")

    config = load_config("configs/config.yaml")

    model, metrics = run_training_pipeline(config)

    log("Pipeline finished successfully!")


if __name__ == "__main__":
    main()