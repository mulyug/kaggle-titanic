from omegaconf import OmegaConf

from src.utils import set_seed


def main():
    config = OmegaConf.load("configs/config.yaml")

    set_seed(config.general.seed)



if __name__ == "__main__":
    main()