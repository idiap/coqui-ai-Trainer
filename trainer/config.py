from dataclasses import dataclass, field
from typing import Any

from coqpit import Coqpit


@dataclass
class TrainerArgs(Coqpit):
    """Trainer arguments that can be accessed from the command line.

    Examples::
        >>> python train.py --restore_path /path/to/checkpoint.pth
    """

    continue_path: str = field(
        default="",
        metadata={
            "help": "Path to a training folder to continue training. Restore the model from the last checkpoint and continue training under the same folder."
        },
    )
    restore_path: str = field(
        default="",
        metadata={
            "help": "Path to a model checkpoit. Restore the model with the given checkpoint and start a new training."
        },
    )
    best_path: str = field(
        default="",
        metadata={
            "help": "Best model file to be used for extracting the best loss. If not specified, the latest best model in continue path is used"
        },
    )
    use_ddp: bool = field(
        default=False,
        metadata={"help": "Use DDP in distributed training. It is to set in `distribute.py`. Do not set manually."},
    )
    use_accelerate: bool = field(default=False, metadata={"help": "Use HF Accelerate as the back end for training."})
    grad_accum_steps: int = field(
        default=1,
        metadata={
            "help": "Number of gradient accumulation steps. It is used to accumulate gradients over multiple batches."
        },
    )
    overfit_batch: bool = field(default=False, metadata={"help": "Overfit a single batch for debugging."})
    skip_train_epoch: bool = field(
        default=False,
        metadata={"help": "Skip training and only run evaluation and test."},
    )
    start_with_eval: bool = field(
        default=False,
        metadata={"help": "Start with evaluation and test."},
    )
    small_run: int | None = field(
        default=None,
        metadata={
            "help": "Only use a subset of the samples for debugging. Set the number of samples to use. Defaults to None. "
        },
    )
    gpu: int | None = field(
        default=None, metadata={"help": "GPU ID to use if ```CUDA_VISIBLE_DEVICES``` is not set. Defaults to None."}
    )
    # only for DDP
    rank: int = field(default=0, metadata={"help": "Process rank in a distributed training. Don't set manually."})
    group_id: str = field(
        default="", metadata={"help": "Process group id in a distributed training. Don't set manually."}
    )


@dataclass
class TrainerConfig(Coqpit):
    """Config fields tweaking the Trainer for a model.

    A ````ModelConfig```, by inheriting ```TrainerConfig``` must be defined for using 👟.
    Inherit this by a new model config and override the fields as needed.
    All the fields can be overridden from comman-line as ```--coqpit.arg_name=value```.

    Example::

        Run the training code by overriding the ```lr``` and ```plot_step``` fields.

        >>> python train.py --coqpit.plot_step=22 --coqpit.lr=0.001

        Defining a model using ```TrainerConfig```.

        >>> from trainer import TrainerConfig
        >>> class MyModelConfig(TrainerConfig):
        ...     optimizer: str = "Adam"
        ...     lr: float = 0.001
        ...     epochs: int = 1
        ...     ...
        >>> class MyModel(nn.module):
        ...    def __init__(self, config):
        ...        ...
        >>> model = MyModel(MyModelConfig())

    """

    # Fields for the run
    output_path: str = field(default="output")
    logger_uri: str | None = field(
        default=None,
        metadata={
            "help": "URI to save training artifacts by the logger. If not set, logs will be saved in the output_path. Defaults to None"
        },
    )
    run_name: str = field(default="run", metadata={"help": "Name of the run. Defaults to 'run'"})
    project_name: str | None = field(default=None, metadata={"help": "Name of the project. Defaults to None"})
    run_description: str = field(
        default="🐸Coqui trainer run.",
        metadata={"help": "Notes and description about the run. Defaults to '🐸Coqui trainer run.'"},
    )
    # Fields for logging
    print_step: int = field(
        default=25, metadata={"help": "Print training stats on the terminal every print_step steps. Defaults to 25"}
    )
    plot_step: int = field(
        default=100, metadata={"help": "Plot training stats on the logger every plot_step steps. Defaults to 100"}
    )
    model_param_stats: bool = field(
        default=False, metadata={"help": "Log model parameters stats on the logger dashboard. Defaults to False"}
    )
    wandb_entity: str | None = field(default=None, metadata={"help": "Wandb entity to log the run. Defaults to None"})
    dashboard_logger: str = field(
        default="tensorboard", metadata={"help": "Logger to use for the tracking dashboard. Defaults to 'tensorboard'"}
    )
    # Fields for checkpointing
    save_on_interrupt: bool = field(
        default=True, metadata={"help": "Save checkpoint on interrupt (Ctrl+C). Defaults to True"}
    )
    log_model_step: int | None = field(
        default=None,
        metadata={
            "help": "Save checkpoint to the logger every log_model_step steps. If not defined `save_step == log_model_step`."
        },
    )
    save_step: int = field(
        default=10000, metadata={"help": "Save local checkpoint every save_step steps. Defaults to 10000"}
    )
    save_n_checkpoints: int = field(default=5, metadata={"help": "Keep n local checkpoints. Defaults to 5"})
    save_checkpoints: bool = field(default=True, metadata={"help": "Save checkpoints locally. Defaults to True"})
    save_all_best: bool = field(
        default=False, metadata={"help": "Save all best checkpoints and keep the older ones. Defaults to False"}
    )
    save_best_after: int = field(default=0, metadata={"help": "Wait N steps to save best checkpoints. Defaults to 0"})
    target_loss: str | None = field(
        default=None, metadata={"help": "Target loss name to select the best model. Defaults to None"}
    )
    # Fields for eval and test run
    print_eval: bool = field(default=False, metadata={"help": "Print eval steps on the terminal. Defaults to False"})
    test_delay_epochs: int = field(default=0, metadata={"help": "Wait N epochs before running the test. Defaults to 0"})
    run_eval: bool = field(
        default=True, metadata={"help": "Run evalulation epoch after training epoch. Defaults to True"}
    )
    run_eval_steps: int | None = field(
        default=None,
        metadata={
            "help": "Run evalulation epoch after N steps. If None, waits until training epoch is completed. Defaults to None"
        },
    )
    # Fields for distributed training
    distributed_backend: str = field(
        default="nccl", metadata={"help": "Distributed backend to use. Defaults to 'nccl'"}
    )
    distributed_url: str = field(
        default="tcp://localhost:54321",
        metadata={"help": "Distributed url to use. Defaults to 'tcp://localhost:54321'"},
    )
    # Fields for training specs
    mixed_precision: bool = field(default=False, metadata={"help": "Use mixed precision training. Defaults to False"})
    precision: str = field(
        default="fp16",
        metadata={
            "help": "Precision to use in mixed precision training. `fp16` for float16 and `bf16` for bfloat16. Defaults to 'f16'"
        },
    )
    epochs: int = field(default=1000, metadata={"help": "Number of epochs to train. Defaults to 1000"})
    batch_size: int = field(default=32, metadata={"help": "Batch size to use. Defaults to 32"})
    eval_batch_size: int = field(default=16, metadata={"help": "Batch size to use for eval. Defaults to 16"})
    grad_clip: float = field(
        default=0.0, metadata={"help": "Gradient clipping value. Disabled if <= 0. Defaults to 0.0"}
    )
    scheduler_after_epoch: bool = field(
        default=True,
        metadata={"help": "Step the scheduler after each epoch else step after each iteration. Defaults to True"},
    )
    # Fields for optimzation
    lr: float | list[float] = field(
        default=0.001, metadata={"help": "Learning rate for each optimizer. Defaults to 0.001"}
    )
    optimizer: str | list[str] | None = field(default=None, metadata={"help": "Optimizer(s) to use. Defaults to None"})
    optimizer_params: dict[str, Any] | list[dict[str, Any]] = field(
        default_factory=dict, metadata={"help": "Optimizer(s) arguments. Defaults to {}"}
    )
    lr_scheduler: str | list[str] | None = field(
        default=None, metadata={"help": "Learning rate scheduler(s) to use. Defaults to None"}
    )
    lr_scheduler_params: dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Learning rate scheduler(s) arguments. Defaults to {}"}
    )
    use_grad_scaler: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable gradient scaler explicitly. It is enabled by default with AMP training. Defaults to False"
        },
    )
    allow_tf32: bool = field(
        default=False,
        metadata={
            "help": "A bool that controls whether TensorFloat-32 tensor cores may be used in matrix multiplications on Ampere or newer GPUs. Default to False."
        },
    )
    cudnn_enable: bool = field(default=True, metadata={"help": "Enable/disable cudnn explicitly. Defaults to True"})
    cudnn_deterministic: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable deterministic cudnn operations. Set this True for reproducibility but it slows down training significantly.  Defaults to False."
        },
    )
    cudnn_benchmark: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable cudnn benchmark explicitly. Set this False if your input size change constantly. Defaults to False"
        },
    )
    training_seed: int = field(
        default=54321,
        metadata={"help": "Global seed for torch, random and numpy random number generator. Defaults to 54321"},
    )
