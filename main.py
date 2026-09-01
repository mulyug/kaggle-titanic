from omegaconf import OmegaConf

from src.data import load_data
from src.utils import set_seed


def main():
    config = OmegaConf.load("configs/config.yaml")

    set_seed(config.general.seed)

    train = load_data(config.data.train_path)
    test = load_data(config.data.test_path)

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")


if __name__ == "__main__":
    main()