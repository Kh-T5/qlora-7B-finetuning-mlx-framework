"""Fixed identifiers and conventions that are not tuneable."""

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATASET_NAME = "databricks/databricks-dolly-15k"

IGNORE_INDEX = -100  # cross-entropy skips labels equal to this

ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

LORA_TARGETS = ("q", "k", "v", "o", "gate", "up", "down")

LAYER_WEIGHT_TEMPLATE = "layer_{index:02d}_{name}.npz"
LAYER_NORM_TEMPLATE = "layer_{index:02d}_{name}_layernorm.npy"
NORM_NAMES = ("input", "post_attention")
