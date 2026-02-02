from tests.utils.mnist import MnistModel, MnistModelConfig
from trainer import Trainer, TrainerArgs


def main():
    """Run `MNIST` model training from scratch or from previous checkpoint."""
    # init args and config
    train_args = TrainerArgs()
    config = MnistModelConfig()

    # init the model from config
    model = MnistModel()

    # init the trainer and 🚀
    trainer = Trainer(
        train_args,
        config,
        model=model,
        train_samples=model.get_data_loader(config, is_eval=False),
        eval_samples=model.get_data_loader(config, is_eval=True),
        parse_command_line_args=True,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
