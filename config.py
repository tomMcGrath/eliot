import dataclasses
from typing import List


@dataclasses.dataclass
class ModelConfig:
    """Configuration for a Transformer model."""
    num_layers: int
    d_model: int
    n_heads: int
    d_head: int
    d_mlp: int
    vocab_size: int
    tokenizer: str
    max_len: int


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for a training run."""
    warmup_steps: int
    post_warmup_steps: int
    lr_max: float
    lr_min: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float


@dataclasses.dataclass
class DatasetConfig:
    """Configuration for a dataset loader."""
    train_dataset: str
    seq_len: int
    batch_size: int
    tokenizer: str


@dataclasses.dataclass
class PrelaunchCheckResult:
    """Results from a prelaunch check."""
    passes: bool
    errs: List[str]


@dataclasses.dataclass
class Config:
    """Configuration for a model and training run."""
    model_config: ModelConfig
    training_config: TrainingConfig
    dataset_config: DatasetConfig

    def passes_prelaunch_checks(self):
        """Check that config passes all prelaunch sanity checks."""
        prelaunch_errs = []

        # Does the model tokenizer match the dataset tokenizer?
        if self.model_config.tokenizer != self.dataset_config.tokenizer:
            prelaunch_errs.append(
                ('Tokenizer mismatch! '
                 f'Model tokenizer {self.model_config.tokenizer} '
                 'does not match dataset tokenizer '
                 f'{self.dataset_config.tokenizer}'))

        # Does the model's max_len meet or exceed the dataset's seq_len?
        if self.model_config.max_len < self.dataset_config.seq_len:
            prelaunch_errs.append(
                ('Sequence length mismatch! '
                 f'Model max sequence length {self.model_config.max_len} '
                 'is shorter than dataset sequence length '
                 f'{self.dataset_config.seq_len}')
            )

        if prelaunch_errs:
            return PrelaunchCheckResult(passes=False, errs=prelaunch_errs)
        
        else:
            return PrelaunchCheckResult(passes=True, errs=[])