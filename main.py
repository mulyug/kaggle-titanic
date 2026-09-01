from omegaconf import OmegaConf

from src.utils import set_seed
from src.data import load_data
from src.features import prepare_features


def main():
    config = OmegaConf.load("configs/config.yaml")

    set_seed(config.general.seed)

    train = load_data(config.data.train_path)
    test = load_data(config.data.test_path)

    X, y = prepare_features(
        train,
        config.target.column,
    )

    X_test = test[X.columns].copy()

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X_test shape: {X_test.shape}")


if __name__ == "__main__":
    main()