import mlx.core as mx
import mlx.nn as nn

from mistral_qlora.model.model import MistralModel
from mistral_qlora.model.model_utils import MistralConfig
from mistral_qlora.train.loss import mean_ce


class MistralForCausalLM(nn.Module):
    """
    LM wrapper around MistralModel.

    - MistralModel: "backbone" that maps (input_ids, masks, caches) -> logits.
    - MistralForCausalLM: adds training logic (label shifting, loss).
    """

    def __init__(self, model: MistralModel):
        super().__init__()
        self.model = model

    @classmethod
    def from_mistral_7b(
        cls, config: MistralConfig, dir_weights_q: str, path_weights: str
    ) -> "MistralForCausalLM":
        """
        Constructor from saved, frozen weights.
        """
        base = MistralModel.from_mistral_7b(config, dir_weights_q, path_weights)
        return cls(base)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        labels: mx.array | None = None,
        caches=None,
        use_lora: dict | bool = False,
    ):
        """
        Inputs:
            - input_ids: Batch of token ids.

            - attention_mask: (B, T),
                1 for real tokens, 0 for padding.
                -> Used for masking loss and or building the attention mask inside the backbone.

            - labels: (B, T),
                Target token ids. If provided, we'll compute the LM loss.
                Convention: same shape as input_ids.
                We do the usual "predict token t from token t-1" shift.

            - caches:
                KV cache tree, one entry per layer (used during generation).
                Pass None during training.

            - use_lora: dict | bool
                Either:
                - a bool: enable/disable LoRA everywhere
                - a dict: fine-grained control
                    {"q": True, "k": True, "v": True, "o": False, ...}

        Returns:
            - logits: (B, T, vocab_size)
            - loss: scalar mx.array | None
            - new_caches: updated KV caches (for generation)
        """

        logits, new_caches = self.model(
            input_ids,
            attention_mask=attention_mask,
            caches=caches,
            use_lora=use_lora,
        )

        loss = None
        if labels is not None:
            loss = mean_ce(logits, labels, attention_mask)

        return logits, loss, new_caches
