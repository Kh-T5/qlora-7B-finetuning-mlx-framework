import mlx.nn as nn
from src.model.mistral_decoder import MistralDecoder, MistralDecoderLayer
from src.model.model_utils import MistralAttention, MistralConfig, MistralMLP
from src.quant.utils_linear import QuantizedLinear
import mlx.core as mx
import numpy as np


class MistralModel(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        *,
        decoder_layer=MistralDecoderLayer,
        attn=MistralAttention,
        mlp=MistralMLP,
        linear_cls=nn.Linear,
    ):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.decoder = MistralDecoder(
            config,
            decoder_layer=decoder_layer,
            attn=attn,
            mlp=mlp,
            linear_cls=linear_cls,
        )
        self.lm_head = linear_cls(config.embed_dim, config.vocab_size)

    @classmethod
    def from_mistral_7b(
        cls, config: MistralConfig, dir_weights_q: str, path_weights: str
    ) -> "MistralModel":
        new_model = cls(config)
        new_model.decoder = MistralDecoder.build_decoder_from_npz(config, dir_weights_q)
        with np.load(path_weights) as data:
            new_model.embed.weight = mx.array(data["embed_np"])
            weights_lm_head = mx.array(data["head_np"])
        new_model.lm_head = QuantizedLinear.convert_4bit(weights_lm_head)

        return new_model

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        caches=None,
        use_lora: dict | bool = False,
    ):
        """
        Inputs:
            - input_ids: (B, T) T being sequence length
            - attention_mask: (B, T) with 1 for tokens to attend, 0 for padding (optional)
            - caches: optional KV cache passed to the decoder (generation purpose)
            - use_lora: dict, selects where to apply LoRA layers in the form of {
                                                "q": False,
                                                "v": True,
                                                "k": True,
                                                "o": False,
                                                ...
                                            }


        Returns:
            - logits (B, T, vocab_size)
            - cache list[dict] of KV cache for each layer
        """

        x = self.embed(input_ids)  # (B, T, D)

        attn_mask = None
        if attention_mask is not None and caches is None:
            # Training case, we not keep track of caches
            B, T = attention_mask.shape

            causal = mx.full((T, T), float("-inf"))
            causal = mx.triu(causal, k=1)
            causal = causal[None, None, :, :]

            pad = (1.0 - attention_mask.astype(mx.float32)) * float("-inf")
            pad = pad[:, None, None, :]

            attn_mask = causal + pad

        x, new_caches = self.decoder(
            x,
            attn_mask=attn_mask,
            caches=caches,
            positions=None,
            use_lora=use_lora,
        )
        logits = self.lm_head(x)

        return logits, new_caches
