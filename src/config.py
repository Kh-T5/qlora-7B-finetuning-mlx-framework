MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MAX_LENGTH = 128  # Max input length
SEED = 42
split_val = 0.1

### Paths
tokenized_ds_path = r"data/dolly_mistral7b_tokenized"
Dataset_dolly = "databricks/databricks-dolly-15k"
mistral_decoder_layers_quant_dir = r"data/quantized_mistral_7b/decoder_mlp_layers/"
mistral_other_layers_quant_path = (
    r"data/quantized_mistral_7b/other_layers/norm_embed_head.npz"
)


### Model Params
eps_rmsnorm = 1e-5

### Training aprams
LoRA_r = 8
alpa = 16
epochs = 20
dropout = 0.0
