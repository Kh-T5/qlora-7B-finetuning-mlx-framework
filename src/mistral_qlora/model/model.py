import mlx.core as mx
import mlx.nn as nn

from mistral_qlora.checkpoint import load_embeddings
from mistral_qlora.config import MistralConfig
from mistral_qlora.model.mistral_decoder import MistralDecoder, MistralDecoderLayer
from mistral_qlora.model.model_utils import (
    MistralAttention,
    MistralMLP,
    resolve_use_lora,
)
from mistral_qlora.quant.utils_linear import Linear, QuantizedLinear


class MistralModel(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        *,
        decoder_layer=MistralDecoderLayer,
        attn=MistralAttention,
        mlp=MistralMLP,
        linear_cls=Linear,
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
        new_model.decoder = MistralDecoder.build_decoder_from_npz(
            config, dir_weights_q, path_weights
        )
        weights = load_embeddings(path_weights)
        new_model.embed.weight = mx.array(weights["embed"], dtype=mx.float16)
        weights_lm_head = mx.array(weights["head"], dtype=mx.float16)
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
        use_lora = resolve_use_lora(use_lora)

        x = self.embed(input_ids).astype(mx.float16)

        attn_mask = None
        if attention_mask is not None and caches is None:
            B, T = attention_mask.shape

            causal = mx.full((T, T), float("-inf"))
            causal = mx.triu(causal, k=1)
            causal = causal[None, None, :, :]

            pad = mx.where(
                attention_mask.astype(mx.bool_),
                mx.array(0.0, dtype=mx.float16),
                mx.array(float("-inf"), dtype=mx.float16),
            )
            pad = pad[:, None, None, :]

            attn_mask = causal + pad

        x, new_caches = self.decoder(
            x,
            attn_mask=attn_mask,
            caches=caches,
            positions=None,
            use_lora=use_lora,
        )
        logits = self.lm_head(x).astype(mx.float32)

        return logits, new_caches
