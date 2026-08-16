"""Frozen configuration objects, constructed in entry points and passed explicitly.

Three concerns are kept apart: `MistralConfig` is the model's shape, `TrainConfig`
the training knobs, and `Paths` the filesystem layout. Nothing reads these from
module state, so a caller can redirect any one of them without touching the others.
"""

from dataclasses import dataclass, field
from pathlib import Path

from mistral_qlora.constants import LORA_TARGETS

DEFAULT_LORA_TARGETS = {
    "q": True,
    "k": True,
    "v": True,
    "o": False,
    "gate": False,
    "up": False,
    "down": False,
}


@dataclass(frozen=True)
class MistralConfig:
    """Architecture of the model.

    `hidden_size_atten` must equal `num_attention_heads * head_dim`, and
    `embed_dim` must match it so the decoder accepts the embedding output.
    `num_key_value_heads` below `num_attention_heads` selects grouped-query
    attention. `lora_true` marks which projections carry adapters.
    """

    vocab_size: int = 32000
    embed_dim: int = 4096
    hidden_size_atten: int = 4096
    hidden_size_mlp: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_layers: int = 32
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e4
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    lora_true: dict = field(default_factory=lambda: dict(DEFAULT_LORA_TARGETS))

    def __post_init__(self):
        if self.hidden_size_atten != self.num_attention_heads * self.head_dim:
            raise ValueError(
                f"hidden_size_atten ({self.hidden_size_atten}) must equal "
                f"num_attention_heads * head_dim "
                f"({self.num_attention_heads} * {self.head_dim})"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a "
                f"multiple of num_key_value_heads ({self.num_key_value_heads})"
            )
        missing = set(LORA_TARGETS) - set(self.lora_true)
        if missing:
            raise ValueError(f"lora_true is missing projections: {sorted(missing)}")


@dataclass(frozen=True)
class TrainConfig:
    """Knobs that change between runs without changing the model."""

    max_length: int = 128
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 1e-4
    eval_every: int = 500
    seed: int = 42
    val_split: float = 0.1


@dataclass(frozen=True)
class Paths:
    """Filesystem layout, rooted at a single directory."""

    root: Path = Path("data")

    @property
    def tokenized_dataset(self) -> Path:
        return self.root / "dolly_mistral7b_tokenized"

    @property
    def quantized_layers(self) -> Path:
        return self.root / "quantized_mistral_7b" / "decoder_mlp_layers"

    @property
    def quantized_other(self) -> Path:
        return (
            self.root / "quantized_mistral_7b" / "other_layers" / "norm_embed_head.npz"
        )

    @property
    def adapters_dir(self) -> Path:
        return self.root / "lora_adapters_mistral_7b"

    @property
    def training_results(self) -> Path:
        return self.root / "training_results"
