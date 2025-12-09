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
embed_dim = 4096
hidden_size_atten = 4096  # Also embedding dim
vocab_size = 32000
rms_norm_eps = 1e-5
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128
rope_theta = 1e4
hidden_size_mlp = 14336
num_layers = 32


### Training aprams
LoRA_r = 8
alpha = 16
epochs = 20
dropout = 0.0
lora_true = {
    "k": True,
    "v": True,
    "q": False,
    "o": False,
    "gate": False,
    "up": False,
    "down": False,
}
